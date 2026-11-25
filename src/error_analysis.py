"""src/error_analysis.py - Phân tích lỗi cho báo cáo/README (chỉ suy luận, không train).

Đọc checkpoint trong checkpoints/ + processed/, tính top-20 trên TEST rồi trả lời:
  - bao nhiêu % user KHÔNG có phim nào trong top-20 (user khó)
  - Recall@20 chia theo mức độ hoạt động của user (user nhiều/ít tương tác)
  - Recall@20 chia theo độ phổ biến của item (phim head vs tail)
  - thiên lệch popularity: điểm phổ biến trung bình của phim được gợi ý
  - độ phủ catalog: bao nhiêu % trong 3.125 phim từng được gợi ý

Chạy:
    venv\\Scripts\\python.exe src\\error_analysis.py
    venv\\Scripts\\python.exe src\\error_analysis.py --seeds 0,1,2   # trung bình 3 seed
Ghi: results/error_analysis.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "demo"))

from demo.recommender import DemoModel  # noqa: E402
from utils import enable_utf8, K  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def log(msg):
    print(msg, flush=True)


def evaluate(model, train_pairs, val_pairs, test_users, test_items, nu, ni, k=K):
    """Trả về (top20, recall_per_user, ndcg_per_user, n_rel_per_user) trên toàn bộ user."""
    scores = (model.user_emb @ model.item_emb.T).astype(np.float32, copy=False).copy()
    hide = np.zeros((nu, ni), dtype=bool)
    hide[train_pairs[0], train_pairs[1]] = True
    hide[val_pairs[0], val_pairs[1]] = True
    scores[hide] = -np.inf
    del hide
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), 1), 1)
    del scores

    hits = np.zeros(nu)
    ndcg = np.zeros(nu)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    n_rel = np.bincount(test_users, minlength=nu)
    order = np.argsort(test_users, kind="stable")
    tu, ti = test_users[order], test_items[order]
    bounds = np.searchsorted(tu, np.arange(nu + 1))
    for u in np.flatnonzero(n_rel):
        want = set(ti[bounds[u]:bounds[u + 1]].tolist())
        got = top[u]
        h = np.fromiter((1.0 if g in want else 0.0 for g in got), dtype=np.float64, count=len(got))
        hits[u] = h.sum()
        idcg = disc[: min(len(want), k)].sum()
        ndcg[u] = (h * disc).sum() / idcg if idcg > 0 else 0.0
    rec = np.divide(hits, n_rel, out=np.zeros(nu), where=n_rel > 0)
    return top, rec, ndcg, n_rel


def analyse(model, data, k=K):
    nu, ni = model.nu, model.ni
    tr = data["train"]
    va = data["val"]
    te = data["test"]
    tu = te["user"].to_numpy(np.int64)
    ti = te["item"].to_numpy(np.int64)

    top, rec, ndcg, n_rel = evaluate(
        model,
        (tr["user"].to_numpy(np.int64), tr["item"].to_numpy(np.int64)),
        (va["user"].to_numpy(np.int64), va["item"].to_numpy(np.int64)),
        tu, ti, nu, ni, k)

    has_target = n_rel > 0

    pop = tr["item"].value_counts()
    pop_count = np.zeros(ni)
    pop_count[pop.index.to_numpy()] = pop.to_numpy()
    # thứ hạng phổ biến: 0 = phổ biến nhất
    pop_rank = np.empty(ni)
    pop_rank[np.argsort(-pop_count)] = np.arange(ni)
    pop_pct = 1.0 - pop_rank / (ni - 1)                 # 1 = phổ biến nhất, 0 = hiếm nhất

    # 1) user không có hit nào
    no_hit = int((has_target & (rec == 0)).sum())
    n_user = int(has_target.sum())

    # 2) theo mức hoạt động của user
    act = np.zeros(nu, dtype=np.int64)
    for u, c in tr.groupby("user").size().items():
        act[int(u)] = int(c)
    med = np.median(act[has_target])
    q = np.zeros(nu, dtype=int)
    q[has_target] = pd.qcut(act[has_target], 4, labels=False, duplicates="drop")
    by_act = []
    for g in range(4):
        m = has_target & (q == g) if has_target.sum() else has_target
        m = has_target & (q == g)
        if m.sum() == 0:
            continue
        by_act.append(dict(nhom=g + 1, n=int(m.sum()), act_min=int(act[m].min()),
                           act_max=int(act[m].max()), recall=float(rec[m].mean()),
                           ndcg=float(ndcg[m].mean())))

    # 3) theo độ phổ biến của item test (head vs tail)
    test_df = pd.DataFrame({"user": tu, "item": ti})
    test_df["pop_pct"] = pop_pct[test_df["item"]]
    test_df["pop_count"] = pop_count[test_df["item"]]
    test_df["quartile"] = pd.qcut(test_df["pop_pct"], 4, labels=False, duplicates="drop")
    rec_by_user = {u: set(top[u]) for u in np.flatnonzero(has_target)}
    test_df["hit"] = [1.0 if i in rec_by_user.get(u, set()) else 0.0
                      for u, i in zip(test_df["user"], test_df["item"])]
    test_df["act"] = act[test_df["user"]]
    test_df["act_q"] = pd.qcut(test_df["act"], 4, labels=False, duplicates="drop")
    by_act_pairs = test_df.groupby("act_q")["hit"].mean()
    by_pop = (test_df.groupby("quartile")["hit"].mean().reset_index()
              .rename(columns={"quartile": "nhom_pop", "hit": "hit_rate"}))
    by_pop["item_pop_trung_binh"] = test_df.groupby("quartile")["pop_count"].mean().values

    # 4) thiên lệch popularity của phim được gợi ý + độ phủ catalog
    rec_items = top[has_target].ravel()
    bias = float(pop_pct[rec_items].mean())
    coverage = float(len(np.unique(rec_items)) / ni)
    tail_share = float((pop_pct[rec_items] < 0.25).mean())

    return dict(
        n_user=n_user, no_hit=no_hit, no_hit_pct=100.0 * no_hit / max(n_user, 1),
        recall=float(rec[has_target].mean()), ndcg=float(ndcg[has_target].mean()),
        hit_at_20_pairs=float(test_df["hit"].mean()),
        by_act=by_act, by_act_pairs=by_act_pairs, by_pop=by_pop,
        bias=bias, coverage=coverage, tail_share=tail_share,
        median_act=int(med),
    )


def main():
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "processed"))
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=os.path.join(RESULTS, "error_analysis.md"))
    args = ap.parse_args()

    data = {name: pd.read_csv(os.path.join(args.data, "mf", f"{name}.csv"))
            for name in ("train", "val", "test")}
    import json
    stats = json.load(open(os.path.join(args.data, "stats.json")))

    models = {}
    for kind, pref in (("MF", "mf_s"), ("LightGCN", "lightgcn_s")):
        rows = []
        for seed in args.seeds.split(","):
            path = os.path.join(args.ckpt_dir, f"{pref}{seed.strip()}.pth")
            if not os.path.exists(path):
                log(f"bỏ qua (thiếu): {path}")
                continue
            rows.append(analyse(DemoModel(ROOT, path), data))
        if rows:
            models[kind] = dict(
                n_user=rows[0]["n_user"], no_hit=rows[0]["no_hit"],
                no_hit_pct=float(np.mean([r["no_hit_pct"] for r in rows])),
                recall=float(np.mean([r["recall"] for r in rows])),
                ndcg=float(np.mean([r["ndcg"] for r in rows])),
                hit_at_20_pairs=float(np.mean([r["hit_at_20_pairs"] for r in rows])),
                bias=float(np.mean([r["bias"] for r in rows])),
                coverage=float(np.mean([r["coverage"] for r in rows])),
                tail_share=float(np.mean([r["tail_share"] for r in rows])),
                by_act=rows[0]["by_act"], by_act_pairs=rows[0]["by_act_pairs"],
                by_pop=rows[0]["by_pop"],
                seeds=len(rows), median_act=rows[0]["median_act"],
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    L = ["# Phân tích lỗi (sinh tự động bởi src/error_analysis.py)", "",
         f"- Dữ liệu: {stats['n_users']} user, {stats['n_items']} item, "
         f"test {stats['n_test']} cặp (split {stats['split']})",
         f"- Số seed dùng: {args.seeds} (trung bình các seed nếu >1)",
         f"- Metric: Recall@{K} / NDCG@{K}, đã che train + val", ""]

    L += ["## Tổng quan", "", "| Model | Recall@20 | NDCG@20 | User không có hit | Tỉ lệ cặp test vào top-20 |",
          "|---|---|---|---|---|"]
    for kind, m in models.items():
        L.append(f"| {kind} | {m['recall']:.4f} | {m['ndcg']:.4f} | "
                 f"{m['no_hit']:,} ({m['no_hit_pct']:.1f}%) | {100*m['hit_at_20_pairs']:.2f}% |")
    L.append("")

    L += ["## Recall theo mức hoạt động của user (chia 4 nhóm theo số tương tác train)", "",
          "| Model | Nhóm | Số user | Khoảng tương tác | Recall@20 | NDCG@20 |", "|---|---|---|---|---|---|"]
    for kind, m in models.items():
        for g in m["by_act"]:
            L.append(f"| {kind} | {g['nhom']} | {g['n']:,} | {g['act_min']}–{g['act_max']} | "
                     f"{g['recall']:.4f} | {g['ndcg']:.4f} |")
    L.append("")

    L += ["## Tỉ lệ CẶP test vào top-20, chia theo mức hoạt động của user", "",
          "(Chỉ số công bằng hơn Recall trung bình: Recall mỗi user bị chia cho số item test của user đó,",
          "nên user ít tương tác thường có Recall cao hơn một cách máy móc.)", "",
          "| Model | Nhóm hoạt động | Tỉ lệ cặp test vào top-20 |", "|---|---|---|"]
    for kind, m in models.items():
        for q, v in m["by_act_pairs"].items():
            L.append(f"| {kind} | {int(q) + 1} | {100*v:.2f}% |")
    L.append("")

    L += ["## Cặp test vào top-20, chia theo độ phổ biến của phim (nhóm 1 = phim hiếm nhất)", "",
          "| Model | Nhóm | Độ phổ biến TB (số user trong train) | Tỉ lệ vào top-20 |",
          "|---|---|---|---|"]
    for kind, m in models.items():
        for _, r in m["by_pop"].iterrows():
            L.append(f"| {kind} | {int(r['nhom_pop']) + 1} | {r['item_pop_trung_binh']:.1f} | "
                     f"{100*r['hit_rate']:.2f}% |")
    L.append("")

    L += ["## Thiên lệch popularity & độ phủ", "",
          "| Model | Điểm phổ biến TB của phim được gợi ý (0–1) | % gợi ý thuộc nhóm hiếm (bottom 25%) | Độ phủ catalog |",
          "|---|---|---|---|"]
    for kind, m in models.items():
        L.append(f"| {kind} | {m['bias']:.3f} | {100*m['tail_share']:.1f}% | {100*m['coverage']:.1f}% |")
    L.append("")
    L += ["Ghi chú: 'điểm phổ biến' = 1 - thứ_hạng_phổ_biến/(n_items-1), 1 là phim phổ biến nhất.",
          "Độ phủ catalog = số phim khác nhau từng xuất hiện trong top-20 / tổng số phim.", ""]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"Đã ghi {args.out}")
    for kind, m in models.items():
        log(f"{kind}: recall {m['recall']:.4f} | no_hit {m['no_hit_pct']:.1f}% | "
            f"bias {m['bias']:.3f} | coverage {100*m['coverage']:.1f}%")


if __name__ == "__main__":
    main()
