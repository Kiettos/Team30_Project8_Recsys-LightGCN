"""Train_mf.py - BPR-MF (baseline) trÃªn MovieLens-1M, dÃ¹ng CHUNG dá»¯ liá»‡u vÃ  metric vá»›i LightGCN.

Cháº¡y tá»« thÆ° má»¥c gá»‘c project:
    python src/model/Train_mf.py --list                                  # xem káº¿ hoáº¡ch thÃ­ nghiá»‡m
    python src/model/Train_mf.py --only A_mf_d64 --epochs 2              # cháº¡y thá»­ 2 epoch (khÃ´ng ghi káº¿t quáº£)
    python src/model/Train_mf.py                                         # HPO: cháº¡y háº¿t thÃ­ nghiá»‡m trong config
    python src/model/Train_mf.py --final                                 # cáº¥u hÃ¬nh tá»‘t nháº¥t x 3 seed -> TEST

VÃ¬ sao BPR-MF? Dá»¯ liá»‡u lÃ  implicit (rating >= 4), bÃ i toÃ¡n lÃ  xáº¿p háº¡ng top-K, nÃªn
baseline pháº£i tá»‘i Æ°u cÃ¹ng má»¥c tiÃªu vá»›i LightGCN (BPR) vÃ  Ä‘Ã¡nh giÃ¡ báº±ng cÃ¹ng
Recall@20 / NDCG@20. Náº¿u dÃ¹ng MF dá»± Ä‘oÃ¡n rating (MSE) thÃ¬ so sÃ¡nh sáº½ khÃ´ng cÃ´ng báº±ng.

MF chÃ­nh lÃ  LightGCN vá»›i 0 táº§ng lan truyá»n: score(u,i) = p_u . q_i.
Káº¿t quáº£: results/hpo_mf.csv, results/final_mf.csv, checkpoints/mf_s*.pth
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from utils import (K, ROOT, RESULTS_DIR, CKPT_DIR, get_device, load_data,  # noqa: E402
                   append_csv, evaluate_ranking, read_csv, resolve, save_checkpoint,
                   train_bpr, enable_utf8)

HPO_CSV = os.path.join(RESULTS_DIR, "hpo_mf.csv")
FINAL_CSV = os.path.join(RESULTS_DIR, "final_mf.csv")
SKIP_KEYS = ("name", "stage", "note")


class MF(nn.Module):
    """BPR-MF: score(u, i) = p_u . q_i (khÃ´ng bias, khÃ´ng lan truyá»n Ä‘á»“ thá»‹)."""

    def __init__(self, nu, ni, dim):
        super().__init__()
        self.user, self.item = nn.Embedding(nu, dim), nn.Embedding(ni, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def batch_embeddings(self, u, pos, neg):
        pu, pp, pn = self.user(u), self.item(pos), self.item(neg)
        return pu, pp, pn, [pu, pp, pn]

    def score_all(self):
        return self.user.weight @ self.item.weight.T


def load_config(path):
    with open(resolve(path), encoding="utf-8") as f:
        raw = json.load(f)
    exps = {}
    for e in raw["experiments"]:
        cfg = dict(raw["base"])
        cfg.update({k: v for k, v in e.items() if k not in SKIP_KEYS})
        cfg["name"] = e["name"]
        exps[e["name"]] = cfg
    return exps, raw.get("final_seeds", [0, 1, 2])


def describe(exps, done):
    print(f"{'name':28s} {'dim':>4s} {'lr':>7s} {'reg':>8s}  tráº¡ng thÃ¡i")
    for name, c in exps.items():
        status = "Ä‘Ã£ cÃ³ káº¿t quáº£" if name in done else "chÆ°a cháº¡y"
        print(f"{name:28s} {c['dim']:4d} {c['lr']:7g} {c['reg']:8g}  {status}")


def main():
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpo_config_mf.json")
    ap.add_argument("--data_dir", default="processed")
    ap.add_argument("--only", default="", help="tÃªn thÃ­ nghiá»‡m, cÃ¡ch nhau báº±ng dáº¥u pháº©y")
    ap.add_argument("--epochs", type=int, default=0,
                    help="ghi Ä‘Ã¨ sá»‘ epoch (cháº¡y thá»­, KHÃ”NG ghi káº¿t quáº£/checkpoint)")
    ap.add_argument("--final", action="store_true", help="cáº¥u hÃ¬nh tá»‘t nháº¥t x cÃ¡c seed -> TEST")
    ap.add_argument("--list", action="store_true", help="in káº¿ hoáº¡ch thÃ­ nghiá»‡m rá»“i thoÃ¡t")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto",
                    help="auto: dÃ¹ng GPU náº¿u cÃ³; cpu: chá»‰ Ä‘á»ƒ cháº¡y thá»­")
    args = ap.parse_args()

    exps, seeds = load_config(args.config)
    if args.list:
        describe(exps, {r["name"] for r in read_csv(HPO_CSV)})
        return

    if args.epochs:
        for c in exps.values():
            c["epochs"] = args.epochs

    dev = get_device(require_gpu=(args.device != "cpu" and not args.epochs))
    if args.device == "cpu":
        dev = torch.device("cpu")

    D = load_data(args.data_dir, dev)
    D["train_u"], D["train_i"] = D["train_u"].to(dev), D["train_i"].to(dev)
    print(f"users {D['nu']} | items {D['ni']} | train {len(D['train_u'])}")

    def make_model(cfg):
        return MF(D["nu"], D["ni"], cfg["dim"]).to(dev)

    # ---------- FINAL: cháº¡y trÃªn TEST ----------
    if args.final:
        rows = read_csv(HPO_CSV)
        if not rows:
            raise SystemExit("ChÆ°a cÃ³ káº¿t quáº£ HPO (results/hpo_mf.csv). HÃ£y cháº¡y HPO trÆ°á»›c.")
        best_name = max(rows, key=lambda r: float(r["val_ndcg20"]))["name"]
        cfg0 = exps[best_name]
        print(f"FINAL MF: {best_name} {cfg0}")
        hide = D["train"] | D["val"]                     # test: che cáº£ train vÃ  val
        res = []
        for seed in seeds:
            cfg = dict(cfg0, seed=seed)
            model, info = train_bpr(make_model(cfg), D, cfg, f"s{seed}")
            tr, tn = evaluate_ranking(model.score_all(), hide, D["test"])
            print(f"  >>> seed {seed}: TEST Recall@{K} {tr:.4f} | TEST NDCG@{K} {tn:.4f}")
            res.append((tr, tn))
            if not args.epochs:
                save_checkpoint(os.path.join(CKPT_DIR, f"mf_s{seed}.pth"), model, cfg,
                                extra=dict(best_epoch=info["best_epoch"]))
                append_csv(FINAL_CSV, dict(name=best_name, seed=seed, dim=cfg["dim"],
                                           lr=cfg["lr"], reg=cfg["reg"],
                                           best_epoch=info["best_epoch"],
                                           test_recall20=f"{tr:.4f}", test_ndcg20=f"{tn:.4f}"))
        if len(res) > 1:
            import numpy as np
            a = np.array(res)
            print(f"\nTEST ({best_name}): Recall@{K} {a[:,0].mean():.4f} Â± {a[:,0].std(ddof=1):.4f} | "
                  f"NDCG@{K} {a[:,1].mean():.4f} Â± {a[:,1].std(ddof=1):.4f}")
        return

    # ---------- HPO: chá»n theo VALIDATION ----------
    names = [n for n in exps if not args.only or n in args.only.split(",")]
    if not names:
        raise SystemExit("KhÃ´ng cÃ³ thÃ­ nghiá»‡m nÃ o khá»›p --only")
    done = {r["name"] for r in read_csv(HPO_CSV)}
    for k, name in enumerate(names, 1):
        if name in done and not args.epochs:
            print(f"[{k}/{len(names)}] {name}: Ä‘Ã£ xong, bá» qua")
            continue
        cfg = exps[name]
        print(f"\n[{k}/{len(names)}] {name}  dim={cfg['dim']} lr={cfg['lr']:g} reg={cfg['reg']:g}")
        model, info = train_bpr(make_model(cfg), D, cfg, name)
        print(f"  -> best epoch {info['best_epoch']} | VAL Recall@{K} {info['val_recall20']:.4f} | "
              f"VAL NDCG@{K} {info['val_ndcg20']:.4f} | {info['seconds']:.0f}s")
        if not args.epochs:
            append_csv(HPO_CSV, dict(name=name, dim=cfg["dim"], lr=cfg["lr"], reg=cfg["reg"],
                                     batch=cfg["batch"], best_epoch=info["best_epoch"],
                                     val_recall20=f"{info['val_recall20']:.4f}",
                                     val_ndcg20=f"{info['val_ndcg20']:.4f}",
                                     seconds=f"{info['seconds']:.0f}"))
    print("\nXong HPO MF. Tiáº¿p theo: python src/model/Train_mf.py --final")


if __name__ == "__main__":
    main()
