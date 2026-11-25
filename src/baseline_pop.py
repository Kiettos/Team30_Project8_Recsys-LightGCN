"""baseline_pop.py - Baseline "MostPop" (gá»£i Ã½ phim phá»• biáº¿n nháº¥t) Ä‘á»ƒ kiá»ƒm tra chÃ©o metric.

ÄÃ¢y khÃ´ng pháº£i mÃ´ hÃ¬nh há»c, chá»‰ lÃ  má»‘c so sÃ¡nh ráº» nháº¥t: láº¥y top-20 item cÃ³ nhiá»u
tÆ°Æ¡ng tÃ¡c train nháº¥t, loáº¡i cÃ¡c item user Ä‘Ã£ xem, rá»“i tÃ­nh Recall@20 / NDCG@20
báº±ng ÄÃšNG hÃ m Ä‘Ã¡nh giÃ¡ dÃ¹ng cho MF vÃ  LightGCN (src/utils.py).

Cháº¡y: python src/baseline_pop.py
Ghi : results/baseline_popularity.csv

Náº¿u MF/LightGCN khÃ´ng vÆ°á»£t Ä‘Æ°á»£c má»‘c nÃ y thÃ¬ cÃ³ lá»—i á»Ÿ dá»¯ liá»‡u hoáº·c á»Ÿ metric.
"""
import os
import sys

import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from utils import K, RESULTS_DIR, append_csv, get_device, load_data, evaluate_ranking, enable_utf8  # noqa: E402

OUT_CSV = os.path.join(RESULTS_DIR, "baseline_popularity.csv")


def main():
    enable_utf8()
    dev = get_device(require_gpu=False)
    D = load_data("processed", dev)
    nu, ni = D["nu"], D["ni"]

    pop = D["train"].sum(0).float()                    # sá»‘ láº§n xuáº¥t hiá»‡n trong train
    scores = pop.unsqueeze(0).expand(nu, ni).contiguous()

    rows = []
    for split, hide, target in (("val", D["train"], D["val"]),
                                ("test", D["train"] | D["val"], D["test"])):
        r, n = evaluate_ranking(scores, hide, target)
        print(f"MostPop  {split:4s} | Recall@{K} {r:.4f} | NDCG@{K} {n:.4f}")
        rows.append(dict(model="MostPop", split=split,
                         recall20=f"{r:.4f}", ndcg20=f"{n:.4f}"))

    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)                             # cháº¡y láº¡i thÃ¬ ghi Ä‘Ã¨, khÃ´ng cá»™ng dá»“n
    for row in rows:
        append_csv(OUT_CSV, row)
    print("ÄÃ£ ghi", OUT_CSV)


if __name__ == "__main__":
    main()
