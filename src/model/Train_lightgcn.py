# hpo_lightgcn.py - hyper-parameter search for LightGCN (and MF = LightGCN with layers=0)
#
#   python hpo_lightgcn.py --list                      # show the plan and what is already done
#   python hpo_lightgcn.py --only A_mf_d64 --epochs 2  # quick smoke test (2 epochs)
#   python hpo_lightgcn.py                             # run all experiments in hpo_config.json
#   python hpo_lightgcn.py --final                     # best MF and best LightGCN x 3 seeds -> TEST once
#
# HPO selects on VALIDATION. The test set is only used by --final.
import argparse, csv, json, os, pickle, time
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn

OUT_DIR = 'processed_mf'
RESULTS_DIR = 'results'
HPO_CSV = os.path.join(RESULTS_DIR, 'hpo_results.csv')
FINAL_CSV = os.path.join(RESULTS_DIR, 'final_results.csv')
K = 20
EVAL_BATCH = 1024
ALLOWED = {'dim': int, 'layers': int, 'lr': float, 'reg': float, 'batch': int,
           'epochs': int, 'patience': int, 'min_delta': float, 'seed': int}
HPO_COLS = ['name', 'stage', 'dim', 'layers', 'lr', 'reg', 'batch', 'seed', 'best_epoch', 'epochs_run',
            'val_recall20', 'val_ndcg20', 'seconds', 'cfg']
FINAL_COLS = ['role', 'name', 'dim', 'layers', 'lr', 'reg', 'batch', 'seed', 'best_epoch',
              'val_ndcg20', 'test_recall20', 'test_ndcg20', 'seconds']


# ======================= config + csv helpers (pure python) =======================
def load_config(path):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    base = raw['base']
    exps, seen = [], set()
    for e in raw['experiments']:
        name = e['name']
        if name in seen:
            raise ValueError(f'duplicate experiment name: {name}')
        seen.add(name)
        unknown = set(e) - set(ALLOWED) - {'name', 'stage', 'note'}
        if unknown:
            raise ValueError(f'{name}: unknown keys {sorted(unknown)}')
        cfg = dict(base)
        cfg.update({k: e[k] for k in e if k in ALLOWED})
        missing = set(ALLOWED) - set(cfg)
        if missing:
            raise ValueError(f'{name}: missing keys {sorted(missing)}')
        cfg = {k: ALLOWED[k](cfg[k]) for k in ALLOWED}
        if cfg['layers'] < 0 or cfg['dim'] <= 0 or cfg['lr'] <= 0 or cfg['epochs'] <= 0 or cfg['batch'] <= 0:
            raise ValueError(f'{name}: invalid value in {cfg}')
        exps.append({'name': name, 'stage': e.get('stage', ''), 'cfg': cfg})
    return exps, [int(s) for s in raw.get('final_seeds', [0, 1, 2])]

def cfg_key(cfg):
    return json.dumps(cfg, sort_keys=True)

def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def append_row(path, cols, row):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)

def pick_best(rows, family, name=''):
    """family = 'mf' (layers == 0) or 'lgcn' (layers >= 1). Best = highest VALIDATION NDCG@20."""
    cand = [r for r in rows if (int(r['layers']) == 0) == (family == 'mf')]
    if name:
        cand = [r for r in cand if r['name'] == name]
    if not cand:
        raise SystemExit(f'No finished HPO row for family={family} name={name!r}. Run the search first.')
    best = max(cand, key=lambda r: float(r['val_ndcg20']))
    return best['name'], json.loads(best['cfg'])


# ======================= data (validation split, graph, masks) =======================
def make_keys(u, i, n_items):
    return u.long() * n_items + i.long()

def in_set(keys, sorted_keys):
    if len(sorted_keys) == 0:
        return torch.zeros_like(keys, dtype=torch.bool)
    pos = torch.searchsorted(sorted_keys, keys).clamp_(max=len(sorted_keys) - 1)
    return sorted_keys[pos] == keys

def load_numpy():
    with open(os.path.join(OUT_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    n_users, n_items = meta['n_users'], meta['n_items']
    test_dict = {int(u): [int(i) for i in items] for u, items in meta['test_dict'].items()}
    u0 = np.load(os.path.join(OUT_DIR, 'train_u.npy'))
    i0 = np.load(os.path.join(OUT_DIR, 'train_i.npy'))
    # validation = last training interaction of each user with >= 2 train items
    # (a time split only if train_u/train_i are in time order inside each user)
    last_idx = {}
    for idx, u in enumerate(u0):
        last_idx[int(u)] = idx
    counts = np.bincount(u0, minlength=n_users)
    val_idx = np.array([i for u, i in last_idx.items() if counts[u] >= 2], dtype=np.int64)
    keep = np.ones(len(u0), dtype=bool)
    keep[val_idx] = False
    val_dict = {int(u0[i]): [int(i0[i])] for i in val_idx}
    return dict(n_users=n_users, n_items=n_items, fit_u=u0[keep], fit_i=i0[keep],
                full_u=u0, full_i=i0, val_dict=val_dict, test_dict=test_dict)

class Prepared:
    def __init__(self, device):
        d = load_numpy()
        self.device = device
        self.n_users, self.n_items = d['n_users'], d['n_items']
        nu, ni = self.n_users, self.n_items

        self.fit_u = torch.from_numpy(d['fit_u']).long().to(device)
        self.fit_i = torch.from_numpy(d['fit_i']).long().to(device)
        self.n_train = len(self.fit_u)
        self.train_keys = torch.sort(make_keys(self.fit_u, self.fit_i, ni)).values   # for negative sampling

        def csr(u, i):
            R = sp.coo_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(nu, ni)).tocsr()
            R.data[:] = 1.0                                   # remove duplicate counts
            return R
        R_fit, R_full = csr(d['fit_u'], d['fit_i']), csr(d['full_u'], d['full_i'])
        to_t = lambda R: (torch.from_numpy(R.indptr).long().to(device), torch.from_numpy(R.indices).long().to(device))
        self.mask_fit = to_t(R_fit)      # val evaluation hides TRAIN items
        self.mask_full = to_t(R_full)    # test evaluation hides TRAIN + VAL items

        # normalized adjacency from TRAIN edges only (val and test edges are never in the graph)
        adj = sp.bmat([[None, R_fit], [R_fit.T, None]]).tocsr()
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=rowsum > 0, out=np.zeros_like(rowsum))
        D = sp.diags(d_inv)
        na = (D @ adj @ D).tocsr().astype(np.float32)
        self.adj = torch.sparse_csr_tensor(torch.from_numpy(na.indptr).long(), torch.from_numpy(na.indices).long(),
                                           torch.from_numpy(na.data), size=na.shape).to(device)

        self.val = self._target(d['val_dict'])
        self.test = self._target(d['test_dict'])
        print(f"users {nu} | items {ni} | train {self.n_train} | val users {len(self.val['users'])} | "
              f"test users {len(self.test['users'])} | device {device}")

    def _target(self, dct):
        users = sorted(dct)
        us, its = [], []
        for u in users:
            us += [u] * len(dct[u]); its += dct[u]
        dev = self.device
        keys = make_keys(torch.tensor(us, device=dev), torch.tensor(its, device=dev), self.n_items)
        return {'users': torch.tensor(users, device=dev), 'keys': torch.sort(keys).values,
                'counts': torch.tensor([len(dct[u]) for u in users], device=dev).float()}


# ======================= model, sampling, evaluation =======================
class LightGCN(nn.Module):
    """layers = 0 -> no propagation -> exactly matrix factorization."""
    def __init__(self, n_users, n_items, dim, n_layers, adj):
        super().__init__()
        self.n_users, self.n_items, self.n_layers, self.adj = n_users, n_items, n_layers, adj
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        outs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            outs.append(x)
        final = torch.stack(outs, dim=1).mean(dim=1)
        return torch.split(final, [self.n_users, self.n_items])

def sample_negatives(P, u):
    neg = torch.randint(0, P.n_items, u.shape, device=P.device)
    bad = in_set(make_keys(u, neg, P.n_items), P.train_keys)
    while bad.any():
        neg[bad] = torch.randint(0, P.n_items, (int(bad.sum()),), device=P.device)
        bad = in_set(make_keys(u, neg, P.n_items), P.train_keys)
    return neg

@torch.no_grad()
def evaluate(model, P, target, mask):
    """Recall@K and NDCG@K (full ranking, already-known items hidden)."""
    model.eval()
    users_emb, items_emb = model()
    indptr, indices = mask
    disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=P.device).float())
    idcg_tab = torch.cumsum(disc, 0)
    rec_sum, ndcg_sum = 0.0, 0.0
    users = target['users']
    for s in range(0, len(users), EVAL_BATCH):
        ub = users[s:s + EVAL_BATCH]
        scores = users_emb[ub] @ items_emb.T
        starts, ends = indptr[ub], indptr[ub + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total > 0:
            rows = torch.repeat_interleave(torch.arange(len(ub), device=P.device), counts)
            offs = torch.arange(total, device=P.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            cols = indices[torch.repeat_interleave(starts, counts) + offs]
            scores[rows, cols] = -1e9
        topk = torch.topk(scores, K, dim=1).indices
        hits = in_set(make_keys(ub[:, None].expand_as(topk), topk, P.n_items), target['keys'])   # (B, K)
        cnt = target['counts'][s:s + EVAL_BATCH]
        rec_sum += (hits.sum(1) / cnt).sum().item()
        dcg = (hits.float() * disc).sum(1)
        idcg = idcg_tab[cnt.clamp(max=K).long() - 1]
        ndcg_sum += (dcg / idcg).sum().item()
    return rec_sum / len(users), ndcg_sum / len(users)


# ======================= training (selected on VALIDATION) =======================
def train_one(cfg, P, label=''):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    if P.device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    model = LightGCN(P.n_users, P.n_items, cfg['dim'], cfg['layers'], P.adj).to(P.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best, best_r, best_ep, bad, best_state, ep = -1.0, 0.0, 0, 0, None, 0
    t0 = time.time()
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(P.n_train, device=P.device)
        total = torch.zeros((), device=P.device)
        for s in range(0, P.n_train, cfg['batch']):
            b = perm[s:s + cfg['batch']]
            u, pos = P.fit_u[b], P.fit_i[b]
            neg = sample_negatives(P, u)
            users_emb, items_emb = model()
            u_e, p_e, n_e = users_emb[u], items_emb[pos], items_emb[neg]
            loss = -torch.log(torch.sigmoid((u_e * p_e).sum(1) - (u_e * n_e).sum(1)) + 1e-8).mean()
            # L2 on the layer-0 embeddings of the batch (same definition as Train_MF.py)
            u0, p0, n0 = model.user_emb(u), model.item_emb(pos), model.item_emb(neg)
            loss = loss + cfg['reg'] * (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) / len(b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.detach() * len(b)
        r, n = evaluate(model, P, P.val, P.mask_fit)
        improved = n > best + cfg['min_delta']
        if improved:
            best, best_r, best_ep, bad = n, r, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [{label}] ep {ep:3d} | loss {total.item() / P.n_train:.4f} | VAL Recall@{K} {r:.4f} | "
              f"VAL NDCG@{K} {n:.4f}{'  *' if improved else ''}")
        if bad >= cfg['patience']:
            print(f'  [{label}] early stop (no improvement for {cfg["patience"]} epochs)')
            break
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_ep, epochs_run=ep, val_recall20=best_r, val_ndcg20=best, seconds=time.time() - t0)


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='hpo_config.json')
    ap.add_argument('--list', action='store_true', help='show the plan and exit')
    ap.add_argument('--only', default='', help='comma-separated experiment names')
    ap.add_argument('--epochs', type=int, default=0, help='override max epochs (smoke test)')
    ap.add_argument('--rerun', action='store_true', help='run again even if already finished')
    ap.add_argument('--final', action='store_true', help='best MF and best LightGCN, final seeds, TEST once')
    ap.add_argument('--pick_mf', default='', help='experiment name to use for MF in --final')
    ap.add_argument('--pick_lgcn', default='', help='experiment name to use for LightGCN in --final')
    args = ap.parse_args()

    exps, final_seeds = load_config(args.config)
    if args.only:
        want = set(args.only.split(','))
        exps = [e for e in exps if e['name'] in want]
    if args.epochs:
        for e in exps:
            e['cfg']['epochs'] = args.epochs
    rows = read_rows(HPO_CSV)
    done = {(r['name'], r['cfg']) for r in rows}

    if args.list:
        print(f"{'name':32s} {'stage':5s} {'dim':>4s} {'L':>2s} {'lr':>7s} {'reg':>7s}  status")
        for e in exps:
            c = e['cfg']
            st = 'done' if (e['name'], cfg_key(c)) in done else 'todo'
            print(f"{e['name']:32s} {e['stage']:5s} {c['dim']:4d} {c['layers']:2d} {c['lr']:7g} {c['reg']:7g}  {st}")
        print(f"{len(exps)} experiments, {sum((e['name'], cfg_key(e['cfg'])) in done for e in exps)} done")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        print('WARNING: CUDA not available, running on CPU (slow).')
    P = Prepared(device)

    if args.final:
        picks = {'MF': pick_best(rows, 'mf', args.pick_mf), 'LightGCN': pick_best(rows, 'lgcn', args.pick_lgcn)}
        summary = {}
        for role, (name, cfg0) in picks.items():
            print(f'\n=== FINAL {role}: {name} {cfg0} ===')
            res = []
            for seed in final_seeds:
                cfg = dict(cfg0, seed=seed)
                if args.epochs:
                    cfg['epochs'] = args.epochs
                model, info = train_one(cfg, P, f'{role} s{seed}')
                tr, tn = evaluate(model, P, P.test, P.mask_full)        # TEST, once per run
                print(f'  >>> {role} seed {seed}: TEST Recall@{K} {tr:.4f} | NDCG@{K} {tn:.4f}')
                torch.save(model.state_dict(), os.path.join(OUT_DIR, f'final_{role}_s{seed}.pth'))
                append_row(FINAL_CSV, FINAL_COLS, dict(role=role, name=name, dim=cfg['dim'], layers=cfg['layers'],
                           lr=cfg['lr'], reg=cfg['reg'], batch=cfg['batch'], seed=seed, best_epoch=info['best_epoch'],
                           val_ndcg20=f"{info['val_ndcg20']:.4f}", test_recall20=f'{tr:.4f}', test_ndcg20=f'{tn:.4f}',
                           seconds=f"{info['seconds']:.0f}"))
                res.append((tr, tn))
            a = np.array(res)
            sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(2)
            summary[role] = (name, a.mean(axis=0), sd)
        print('\n=== TEST results (mean ± std over seeds) ===')
        for role, (name, m, sd) in summary.items():
            print(f'{role:9s} ({name}): Recall@{K} {m[0]:.4f} ± {sd[0]:.4f} | NDCG@{K} {m[1]:.4f} ± {sd[1]:.4f}')
        return

    todo = [e for e in exps if args.rerun or (e['name'], cfg_key(e['cfg'])) not in done]
    print(f'{len(todo)} experiment(s) to run, {len(exps) - len(todo)} skipped (already done)')
    for n, e in enumerate(todo, 1):
        c = e['cfg']
        print(f"\n[{n}/{len(todo)}] {e['name']}  dim={c['dim']} layers={c['layers']} lr={c['lr']:g} reg={c['reg']:g}")
        model, info = train_one(c, P, e['name'])
        row = dict(name=e['name'], stage=e['stage'], dim=c['dim'], layers=c['layers'], lr=c['lr'], reg=c['reg'],
                   batch=c['batch'], seed=c['seed'], best_epoch=info['best_epoch'], epochs_run=info['epochs_run'],
                   val_recall20=f"{info['val_recall20']:.4f}", val_ndcg20=f"{info['val_ndcg20']:.4f}",
                   seconds=f"{info['seconds']:.0f}", cfg=cfg_key(c))
        append_row(HPO_CSV, HPO_COLS, row)
        print(f"  -> best epoch {info['best_epoch']} | VAL Recall@{K} {info['val_recall20']:.4f} | "
              f"VAL NDCG@{K} {info['val_ndcg20']:.4f} | {info['seconds']:.0f}s")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    rows = read_rows(HPO_CSV)
    print('\n=== HPO leaderboard (validation NDCG@20) ===')
    for r in sorted(rows, key=lambda r: -float(r['val_ndcg20']))[:10]:
        print(f"{r['name']:32s} L={r['layers']} d={r['dim']:>3s} lr={float(r['lr']):g} reg={float(r['reg']):g} "
              f"| VAL Recall@{K} {r['val_recall20']} NDCG@{K} {r['val_ndcg20']} | epoch {r['best_epoch']}")
    print('\nNext: python hpo_lightgcn.py --final')

if __name__ == '__main__':
    main()