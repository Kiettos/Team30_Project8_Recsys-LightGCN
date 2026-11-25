"""utils.py - Hàm dùng chung cho MF và LightGCN.

Mục tiêu: hai mô hình dùng CHUNG một cách đọc dữ liệu, một hàm đánh giá và một
định dạng CSV, để bảng so sánh trong báo cáo là công bằng (cùng split, cùng
metric, cùng top-K).

Chạy từ thư mục gốc project nên các đường dẫn mặc định đều tính từ ROOT.
"""
import csv
import json
import os
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:                  # thông báo dễ hiểu thay vì traceback
    raise SystemExit(
        "Thiếu PyTorch. Cài môi trường theo README mục 'Cài đặt':\n"
        "  python -m pip install -r requirements.txt"
    )

K = 20                                   # metric chính: Recall@20 / NDCG@20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "checkpoints")


def resolve(path):
    """Đường dẫn tương đối -> ưu tiên CWD, sau đó tới thư mục gốc project."""
    if os.path.isabs(path):
        return path
    return path if os.path.exists(path) else os.path.join(ROOT, path)


def enable_utf8():
    """Console Windows mặc định cp1252/cp1258: bật UTF-8 để in tiếng Việt không crash."""
    import sys
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            if sys.stdout.isatty() and k.GetConsoleOutputCP() != 65001:
                k.SetConsoleOutputCP(65001)      # đổi code page của cửa sổ console
                k.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ------------------------------- seed / device -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_gpu=False):
    """Trả về torch.device. require_gpu=True -> thoát nếu không có CUDA."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        dev = torch.device("cuda")
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
        return dev
    if require_gpu:
        raise SystemExit(
            "Không tìm thấy CUDA. Cài bản PyTorch CUDA (xem README mục 'Cài đặt'), "
            "hoặc chạy với --device cpu để thử code (rất chậm, không dùng để lấy kết quả)."
        )
    print("Device: CPU  (chỉ nên dùng để chạy thử; train thật cần GPU)")
    return torch.device("cpu")


# ------------------------------- dữ liệu -------------------------------
def load_stats(data_dir):
    with open(os.path.join(resolve(data_dir), "stats.json")) as f:
        return json.load(f)


def read_split(data_dir, name):
    """Đọc processed/mf/<name>.csv -> list cột user, item (mảng numpy int64)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(resolve(data_dir), "mf", name + ".csv"))
    return df


def to_mask(u, i, nu, ni, device):
    """Ma trận bool (nu x ni) đánh dấu các cặp (u, i)."""
    m = torch.zeros(nu, ni, dtype=torch.bool, device=device)
    m[torch.as_tensor(u, device=device), torch.as_tensor(i, device=device)] = True
    return m


def load_data(data_dir, device):
    """Đọc train/val/test + dựng mask, dùng chung cho MF và LightGCN.

    Trả về dict:
        nu, ni            : số user / item
        train_u, train_i  : tensor (N,) các cặp train (để lấy batch)
        train, val, test  : mask bool (nu x ni) - đã ở trên device
    """
    stats = load_stats(data_dir)
    nu, ni = stats["n_users"], stats["n_items"]
    tr, va, te = (read_split(data_dir, s) for s in ("train", "val", "test"))
    # copy=True: pandas 3 trả mảng chỉ-đọc, torch cần mảng ghi được
    u = torch.as_tensor(tr["user"].to_numpy(np.int64, copy=True))
    i = torch.as_tensor(tr["item"].to_numpy(np.int64, copy=True))
    return dict(
        nu=nu, ni=ni, stats=stats,
        train_u=u, train_i=i,
        train=to_mask(u, i, nu, ni, device),
        val=to_mask(va["user"].to_numpy(np.int64, copy=True),
                    va["item"].to_numpy(np.int64, copy=True), nu, ni, device),
        test=to_mask(te["user"].to_numpy(np.int64, copy=True),
                     te["item"].to_numpy(np.int64, copy=True), nu, ni, device),
    )


# ------------------------------- đánh giá -------------------------------
@torch.no_grad()
def evaluate_ranking(scores, hide, target, k=K):
    """Recall@K / NDCG@K cho full-ranking.

    scores : (nu x ni) điểm dự đoán (float, trên cùng device)
    hide   : mask bool (nu x ni) các item đã biết -> bị loại khỏi bảng xếp hạng
    target : mask bool (nu x ni) item cần dự đoán (val hoặc test)
    Chỉ tính trên user có ít nhất 1 item trong target.
    """
    scores = scores.clone()
    scores[hide] = -1e9
    top = scores.topk(k, dim=1).indices
    hits = target.gather(1, top).float()                    # (nu, k)
    n_rel = target.sum(1)
    ok = n_rel > 0
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    idcg = disc.cumsum(0)[n_rel.clamp(1, k) - 1]
    recall = hits.sum(1) / n_rel.clamp(min=1)
    ndcg = (hits * disc).sum(1) / idcg
    return recall[ok].mean().item(), ndcg[ok].mean().item()


# ------------------------------- CSV -------------------------------
def append_csv(path, row):
    """Ghi 1 dòng vào CSV, tự tạo header. Dùng để lưu kết quả từng thí nghiệm."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_bpr(model, D, cfg, label="", log=print):
    """Vòng train BPR dùng chung cho MF và LightGCN.

    Model cần cài 2 hàm:
        batch_embeddings(u, pos, neg) -> (pu, pp, pn, raw_list)
            pu/pp/pn  : embedding dùng để tính điểm (đã lan truyền với LightGCN)
            raw_list  : các embedding gốc để tính L2 (giống công thức trong paper)
        score_all() -> (nu x ni) ma trận điểm cho full-ranking

    Luôn chọn checkpoint theo VALIDATION (NDCG@20), early stopping theo patience.
    Trả về (model đã load best state, info dict).
    """
    import time

    set_seed(cfg["seed"])
    dev = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n = len(D["train_u"])
    best_ndcg, best_recall, best_ep, best_state, bad = -1.0, 0.0, 0, None, 0
    t0 = time.time()

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        total = torch.zeros((), device=dev)
        for s in range(0, n, cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            u, pos = D["train_u"][b], D["train_i"][b]
            neg = sample_negatives(u, D)
            pu, pp, pn, raw = model.batch_embeddings(u, pos, neg)
            # -log sigmoid(s_pos - s_neg) == softplus(s_neg - s_pos)
            loss = torch.nn.functional.softplus((pu * pn).sum(1) - (pu * pp).sum(1)).mean()
            l2 = sum(r.pow(2).sum() for r in raw) / len(u)
            loss = loss + cfg["reg"] * l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(b)

        r, nd = evaluate_ranking(model.score_all(), D["train"], D["val"])
        improved = nd > best_ndcg + cfg["min_delta"]
        if improved:
            best_ndcg, best_recall, best_ep, bad = nd, r, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"  [{label}] ep {ep:3d}/{cfg['epochs']} | loss {total.item() / n:.4f} | "
            f"VAL Recall@{K} {r:.4f} | VAL NDCG@{K} {nd:.4f}{' *' if improved else ''}")
        if bad >= cfg["patience"]:
            log(f"  [{label}] early stop")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    info = dict(best_epoch=best_ep, val_recall20=best_recall, val_ndcg20=best_ndcg,
                seconds=time.time() - t0)
    return model, info


@torch.no_grad()
def sample_negatives(u, D):
    """Lấy negative ngẫu nhiên, bốc lại nếu trúng item user đã tương tác trong train."""
    ni, train = D["ni"], D["train"]
    neg = torch.randint(0, ni, u.shape, device=u.device)
    bad = train[u, neg]
    while bad.any():
        neg[bad] = torch.randint(0, ni, (int(bad.sum()),), device=u.device)
        bad = train[u, neg]
    return neg


def save_checkpoint(path, model, cfg, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict(), "cfg": cfg}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
