# src/models/mf.py
import pickle, numpy as np, torch, torch.nn as nn
import os

torch.manual_seed(0); np.random.seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIM, LR, BATCH, EPOCHS, K, REG = 64, 1e-3, 2048, 50, 20, 1e-4

OUT_DIR = 'processed_mf'

with open(os.path.join(OUT_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
    
n_users, n_items = meta['n_users'], meta['n_items']
train_dict, test_dict = meta['train_dict'], meta['test_dict']
train_u = np.load(os.path.join(OUT_DIR, 'train_u.npy'))
train_i = np.load(os.path.join(OUT_DIR, 'train_i.npy'))

class MF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def score(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

    def full_scores(self):
        return self.user_emb.weight @ self.item_emb.weight.T

def sample_negatives(u_batch, pos_batch):
    neg = np.random.randint(0, n_items, size=len(pos_batch))
    for k in range(len(neg)):
        while neg[k] in train_dict.get(u_batch[k], ()):
            neg[k] = np.random.randint(0, n_items)
    return neg

@torch.no_grad()
def evaluate(model, eval_dict, K=20):
    model.eval()
    users = list(eval_dict.keys())
    scores = model.full_scores()[users].clone()
    for r, u in enumerate(users):
        idx = list(train_dict.get(u, ()))
        if idx:
            scores[r, idx] = -1e9
    topk = torch.topk(scores, K, dim=1).indices.cpu().numpy()
    rec, ndcg = [], []
    for r, u in enumerate(users):
        gt = set(eval_dict[u])
        hits = np.isin(topk[r], list(gt))
        rec.append(hits.sum() / len(gt))
        pos = np.where(hits)[0]
        ndcg.append((1.0 / np.log2(pos[0] + 2)) if len(pos) else 0.0)
    return float(np.mean(rec)), float(np.mean(ndcg))

model = MF(n_users, n_items, DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
n_train = len(train_u)

best_ndcg = 0.0  # <--- Track the best score

for ep in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(n_train)
    total = 0.0
    
    for s in range(0, n_train, BATCH):
        b = perm[s:s + BATCH]
        u = train_u[b]
        pos = train_i[b]
        neg = sample_negatives(u, pos)

        u_t   = torch.as_tensor(u,   device=DEVICE)
        pos_t = torch.as_tensor(pos, device=DEVICE)
        neg_t = torch.as_tensor(neg, device=DEVICE)

        s_pos = model.score(u_t, pos_t)
        s_neg = model.score(u_t, neg_t)
        loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()

        reg = (model.user_emb(u_t).pow(2).sum()
             + model.item_emb(pos_t).pow(2).sum()
             + model.item_emb(neg_t).pow(2).sum()) / len(b)
        loss = loss + REG * reg

        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(b)

    if ep % 5 == 0 or ep == 1:
        r, n = evaluate(model, test_dict, K)
        print(f'epoch {ep:3d}  loss {total/n_train:.4f}  '
              f'Recall@{K} {r:.4f}  NDCG@{K} {n:.4f}')
        
        #---Save Best Model
        if n > best_ndcg:
            best_ndcg = n
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'mf_model_best.pth'))
            print(f"   --> 🌟 New best model saved at epoch {ep} (NDCG@{K}: {n:.4f})")

# ---Save Last Model (outside the loop, after epoch 50)
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'mf_model_last.pth'))
print("Finished training. Last model saved.")