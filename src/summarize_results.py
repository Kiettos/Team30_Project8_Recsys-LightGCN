"""summarize_results.py - Gom kết quả train thành bảng so sánh MF vs LightGCN.

Chạy bất cứ lúc nào (không cần train lại), script sẽ báo phần nào còn thiếu:
    python src/summarize_results.py

Đọc (nếu có):
    results/hpo_mf.csv            kết quả HPO của baseline MF
    results/hpo_lightgcn.csv      kết quả HPO của LightGCN
    results/final_mf.csv          kết quả TEST của MF (3 seed)
    results/final_lightgcn.csv    kết quả TEST của LightGCN (3 seed)
    results/baseline_popularity.csv   baseline MostPop (nếu đã chạy)
Ghi:
    results/comparison.md   bảng markdown để dán vào báo cáo/README
    results/comparison.csv  cùng nội dung, dạng máy đọc
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODELS = ("MF", "LightGCN")


def read_csv(path):
    if not os.path.exists(path):
        return []
    # utf-8-sig: chấp nhận cả file có BOM (do Excel/PowerShell sinh ra)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return None, None
    m = sum(values) / len(values)
    sd = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return m, sd


def main():
    paths = {m: (os.path.join(RESULTS, f"hpo_{m.lower()}.csv"),
                 os.path.join(RESULTS, f"final_{m.lower()}.csv")) for m in MODELS}
    lines = ["# Kết quả thực nghiệm", "",
             "Metric: Recall@20 / NDCG@20, full ranking, đã che item user đã biết.",
             "HPO chọn cấu hình theo VALIDATION; TEST chỉ chạy 1 lần với 3 seed.", ""]
    out = []
    any_final = False

    for model in MODELS:
        hpo_path, final_path = paths[model]
        hpo, final = read_csv(hpo_path), read_csv(final_path)
        print(f"\n=== {model} ===")

        if hpo:
            best = max(hpo, key=lambda r: float(r["val_ndcg20"]))
            print(f"HPO: {len(hpo)} thí nghiệm đã chạy; tốt nhất theo VAL: {best['name']} "
                  f"(VAL NDCG@20 = {best['val_ndcg20']})")
            lines += [f"## {model}", "",
                      f"- Số thí nghiệm HPO: {len(hpo)}",
                      f"- Cấu hình tốt nhất theo validation: `{best['name']}` "
                      f"(dim={best.get('dim', '?')}, layers={best.get('layers', '?')}, "
                      f"lr={best.get('lr', '?')}, reg={best.get('reg', '?')})", ""]
        else:
            print(f"HPO: chưa có {os.path.relpath(hpo_path, ROOT)} (chưa chạy HPO)")
            lines += [f"## {model}", "", f"- Chưa có kết quả HPO (`{os.path.relpath(hpo_path, ROOT)}`).", ""]

        if final:
            any_final = True
            m_r, s_r = mean_std([r["test_recall20"] for r in final])
            m_n, s_n = mean_std([r["test_ndcg20"] for r in final])
            print(f"TEST: Recall@20 {m_r:.4f} ± {s_r:.4f} | NDCG@20 {m_n:.4f} ± {s_n:.4f} "
                  f"({len(final)} seed)")
            lines += [f"| Seed | Recall@20 | NDCG@20 |", "|---|---|---|"]
            for r in final:
                lines.append(f"| {r['seed']} | {r['test_recall20']} | {r['test_ndcg20']} |")
            lines += [f"| **TB ± SD** | **{m_r:.4f} ± {s_r:.4f}** | **{m_n:.4f} ± {s_n:.4f}** |", ""]
            for metric, val, sd in (("Recall@20", m_r, s_r), ("NDCG@20", m_n, s_n)):
                out.append(dict(model=model, config=best["name"] if hpo else "",
                                metric=metric, mean=f"{val:.4f}", std=f"{sd:.4f}",
                                seeds=len(final), split="test"))
        else:
            print(f"TEST: chưa có {os.path.relpath(final_path, ROOT)} (chưa chạy --final)")

    pop = read_csv(os.path.join(RESULTS, "baseline_popularity.csv"))
    if pop:
        lines += ["## Baseline tham chiếu", "", "| Model | Split | Recall@20 | NDCG@20 |",
                  "|---|---|---|---|"]
        for r in pop:
            lines.append(f"| {r.get('model', 'MostPop')} | {r.get('split', '')} | "
                         f"{r.get('recall20', '')} | {r.get('ndcg20', '')} |")
            out.append(dict(model=r.get("model", "MostPop"), config="popularity",
                            metric="recall20", mean=r.get("recall20", ""), std="",
                            seeds=1, split=r.get("split", "")))
        lines.append("")

    os.makedirs(RESULTS, exist_ok=True)
    md_path = os.path.join(RESULTS, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nĐã ghi {os.path.relpath(md_path, ROOT)}")

    if out:
        csv_path = os.path.join(RESULTS, "comparison.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"Đã ghi {os.path.relpath(csv_path, ROOT)}")

    if not any_final:
        print("\nChưa có kết quả TEST. Chạy HPO rồi --final cho từng mô hình "
              "(xem README mục 'Cách chạy').")


if __name__ == "__main__":
    main()
