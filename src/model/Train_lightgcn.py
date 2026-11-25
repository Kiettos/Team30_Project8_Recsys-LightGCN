"""train_lightgcn.py - LightGCN trên MovieLens-1M (dữ liệu từ preprocess_ml1m.py).

Chạy từ thư mục gốc project:
    python src/train_lightgcn.py --only A_lgcn_L2_d64 --epochs 2   # chạy thử 2 epoch (không ghi kết quả)
    python src/train_lightgcn.py                                    # chạy hết các thí nghiệm trong config
    python src/train_lightgcn.py --final                            # cấu hình tốt nhất x 3 seed -> TEST
HPO chọn theo VALIDATION. Test chỉ dùng trong --final.
"""
import argparse, csv, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

# Console Windows mặc định cp1252/cp1258 -> tránh UnicodeEncodeError khi in tiếng Việt
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

K = 20
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HPO_CSV, FINAL_CSV, CKPT_DIR = 'results/hpo_lightgcn.csv', 'results/final_lightgcn.csv', 'checkpoints'


# ------------------------------- dữ liệu -------------------------------
def load_data(data_dir):
    stats = json.load(open(os.path.join(data_dir, 'stats.json')))
    nu, ni = stats['n_users'], stats['n_items']

    def read(name):
        df = pd.read_csv(os.path.join(data_dir, 'mf', name + '.csv'))
        return torch.tensor(df['user'].values), torch.tensor(df['item'].values)

    def to_mat(u, i):                       # ma trận bool (n_users x n_items)
        m = torch.zeros(nu, ni, dtype=torch.bool)
        m[u, i] = True
        return m.to(dev)

    (tu, ti), (vu, vi), (eu, ei) = read('train'), read('val'), read('test')
    return dict(nu=nu, ni=ni, u=tu.to(dev), i=ti.to(dev),
                train=to_mat(tu, ti), val=to_mat(vu, vi), test=to_mat(eu, ei))


def build_graph(u, i, nu, ni):
    """R_hat[u,i] = 1/sqrt(deg_u * deg_i), chỉ từ cạnh TRAIN. Trả về R_hat và chuyển vị của nó."""
    du = torch.bincount(u, minlength=nu).float()
    di = torch.bincount(i, minlength=ni).float()
    w = (du[u] * di[i]).rsqrt()
    R = torch.sparse_coo_tensor(torch.stack([u, i]), w, (nu, ni)).coalesce()
    Rt = torch.sparse_coo_tensor(torch.stack([i, u]), w, (ni, nu)).coalesce()
    return R, Rt


# ------------------------------- mô hình -------------------------------
class LightGCN(nn.Module):
    def __init__(self, nu, ni, dim, layers, R, Rt):
        super().__init__()
        self.layers, self.R, self.Rt = layers, R, Rt
        self.user, self.item = nn.Embedding(nu, dim), nn.Embedding(ni, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def propagate(self):
        eu, ei = self.user.weight, self.item.weight
        us, its = [eu], [ei]
        for _ in range(self.layers):
            eu, ei = torch.sparse.mm(self.R, ei), torch.sparse.mm(self.Rt, eu)
            us.append(eu); its.append(ei)
        return torch.stack(us).mean(0), torch.stack(its).mean(0)     # trung bình các tầng


def bpr_loss(model, u, pos, neg, reg):
    eu, ei = model.propagate()
    pu, pp, pn = eu[u], ei[pos], ei[neg]
    loss = F.softplus((pu * pn).sum(1) - (pu * pp).sum(1)).mean()     # -log sigmoid(pos - neg)
    l2 = (model.user(u).pow(2).sum() + model.item(pos).pow(2).sum() + model.item(neg).pow(2).sum()) / len(u)
    return loss + reg * l2


def sample_neg(u, train_mat, ni):
    neg = torch.randint(0, ni, u.shape, device=dev)
    bad = train_mat[u, neg]
    while bad.any():                                                  # bốc lại nếu trúng item đã xem
        neg[bad] = torch.randint(0, ni, (int(bad.sum()),), device=dev)
        bad = train_mat[u, neg]
    return neg


@torch.no_grad()
def evaluate(model, hide, target):
    """Recall@20 và NDCG@20. hide: item đã biết (bị che); target: item cần dự đoán."""
    model.eval()
    eu, ei = model.propagate()
    scores = eu @ ei.T
    scores[hide] = -1e9
    top = scores.topk(K, dim=1).indices
    hits = target.gather(1, top).float()                              # (n_users, K)
    n_rel = target.sum(1)
    ok = n_rel > 0
    disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=dev).float())
    idcg = disc.cumsum(0)[n_rel.clamp(1, K) - 1]
    recall = hits.sum(1) / n_rel.clamp(min=1)
    ndcg = (hits * disc).sum(1) / idcg
    return recall[ok].mean().item(), ndcg[ok].mean().item()


# ------------------------------- huấn luyện -------------------------------
def train(cfg, D, graph, label=''):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model = LightGCN(D['nu'], D['ni'], cfg['dim'], cfg['layers'], *graph).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best_ndcg, best_recall, best_ep, state, bad, t0 = -1.0, 0.0, 0, None, 0, time.time()
    n = len(D['u'])
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        total = torch.zeros((), device=dev)
        for s in range(0, n, cfg['batch']):
            b = perm[s:s + cfg['batch']]
            u, pos = D['u'][b], D['i'][b]
            neg = sample_neg(u, D['train'], D['ni'])
            loss = bpr_loss(model, u, pos, neg, cfg['reg'])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.detach() * len(b)
        r, nd = evaluate(model, D['train'], D['val'])                 # đánh giá trên VALIDATION
        improved = nd > best_ndcg + cfg['min_delta']
        if improved:
            best_ndcg, best_recall, best_ep, bad = nd, r, ep, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [{label}] ep {ep:3d} | loss {total.item() / n:.4f} | VAL Recall@{K} {r:.4f} | "
              f"VAL NDCG@{K} {nd:.4f}{' *' if improved else ''}")
        if bad >= cfg['patience']:
            print(f"  [{label}] early stop")
            break
    model.load_state_dict(state)
    return model, dict(best_epoch=best_ep, val_recall20=best_recall, val_ndcg20=best_ndcg, seconds=time.time() - t0)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/hpo_config_lgcn.json')
    ap.add_argument('--data_dir', default='processed')
    ap.add_argument('--only', default='', help='tên thí nghiệm, cách nhau bằng dấu phẩy')
    ap.add_argument('--epochs', type=int, default=0, help='ghi đè số epoch (chạy thử, không ghi kết quả)')
    ap.add_argument('--final', action='store_true', help='cấu hình tốt nhất x các seed -> TEST')
    args = ap.parse_args()

    raw = json.load(open(args.config, encoding='utf-8'))
    skip = ('name', 'stage', 'note')
    all_exps = {e['name']: {**raw['base'], **{k: v for k, v in e.items() if k not in skip}}
                for e in raw['experiments']}
    seeds = raw.get('final_seeds', [0, 1, 2])
    if args.epochs:
        for c in all_exps.values():
            c['epochs'] = args.epochs

    print('Device:', dev)
    D = load_data(args.data_dir)
    graph = build_graph(D['u'], D['i'], D['nu'], D['ni'])
    print(f"users {D['nu']} | items {D['ni']} | train {len(D['u'])}")

    # ---------- FINAL: chạy trên TEST ----------
    if args.final:
        rows = read_csv(HPO_CSV)
        if not rows:
            raise SystemExit('Chưa có kết quả HPO. Hãy chạy HPO trước.')
        best_name = max(rows, key=lambda r: float(r['val_ndcg20']))['name']
        cfg0 = all_exps[best_name]
        print(f'FINAL: {best_name} {cfg0}')
        hide = D['train'] | D['val']                                  # test: che cả train và val
        res = []
        for seed in seeds:
            cfg = dict(cfg0, seed=seed)
            model, info = train(cfg, D, graph, f's{seed}')
            tr, tn = evaluate(model, hide, D['test'])
            print(f'  >>> seed {seed}: TEST Recall@{K} {tr:.4f} | NDCG@{K} {tn:.4f}')
            res.append((tr, tn))
            if not args.epochs:
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'cfg': cfg}, f'{CKPT_DIR}/lightgcn_s{seed}.pth')
                append_csv(FINAL_CSV, dict(name=best_name, seed=seed, dim=cfg['dim'], layers=cfg['layers'],
                                           lr=cfg['lr'], reg=cfg['reg'], best_epoch=info['best_epoch'],
                                           test_recall20=f'{tr:.4f}', test_ndcg20=f'{tn:.4f}'))
        a = np.array(res)
        m, sd = a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros(2))
        print(f'\nTEST ({best_name}): Recall@{K} {m[0]:.4f} ± {sd[0]:.4f} | NDCG@{K} {m[1]:.4f} ± {sd[1]:.4f}')
        return

    # ---------- HPO: chọn theo VALIDATION ----------
    names = [n for n in all_exps if not args.only or n in args.only.split(',')]
    if not names:
        raise SystemExit('Không có thí nghiệm nào khớp --only')
    done = {r['name'] for r in read_csv(HPO_CSV)}
    for k, name in enumerate(names, 1):
        if name in done and not args.epochs:
            print(f'[{k}/{len(names)}] {name}: đã xong, bỏ qua')
            continue
        cfg = all_exps[name]
        print(f"\n[{k}/{len(names)}] {name}  dim={cfg['dim']} layers={cfg['layers']} lr={cfg['lr']:g} reg={cfg['reg']:g}")
        model, info = train(cfg, D, graph, name)
        print(f"  -> best epoch {info['best_epoch']} | VAL Recall@{K} {info['val_recall20']:.4f} | "
              f"VAL NDCG@{K} {info['val_ndcg20']:.4f} | {info['seconds']:.0f}s")
        if not args.epochs:
            append_csv(HPO_CSV, dict(name=name, dim=cfg['dim'], layers=cfg['layers'], lr=cfg['lr'], reg=cfg['reg'],
                                     batch=cfg['batch'], best_epoch=info['best_epoch'],
                                     val_recall20=f"{info['val_recall20']:.4f}",
                                     val_ndcg20=f"{info['val_ndcg20']:.4f}", seconds=f"{info['seconds']:.0f}"))
    print('\nXong. Tiếp theo: python src/train_lightgcn.py --final')


if __name__ == '__main__':
    main()