# results/ - nơi ghi kết quả thực nghiệm

Thư mục này do các script tự tạo/ghi. Không sửa tay, chỉ đọc.

| File | Sinh ra bởi | Nội dung |
|---|---|---|
| `hpo_mf.csv` | `python src/model/Train_mf.py` | từng thí nghiệm HPO của baseline MF: cấu hình, best_epoch, VAL Recall@20 / NDCG@20, thời gian |
| `hpo_lightgcn.csv` | `python src/model/Train_lightgcn.py` | như trên, cho LightGCN |
| `final_mf.csv` | `python src/model/Train_mf.py --final` | TEST Recall@20 / NDCG@20 cho 3 seed của cấu hình tốt nhất |
| `final_lightgcn.csv` | `python src/model/Train_lightgcn.py --final` | như trên, cho LightGCN |
| `baseline_popularity.csv` | `python src/baseline_pop.py` | mốc MostPop (top-20 phim phổ biến) trên val/test |
| `comparison.md` | `python src/summarize_results.py` | bảng so sánh dán vào báo cáo/README |
| `comparison.csv` | `python src/summarize_results.py` | bảng so sánh dạng máy đọc |

Quy ước: HPO chỉ chọn theo VALIDATION; TEST chỉ được chạy một lần bằng `--final`
với 3 seed (0, 1, 2), báo cáo dạng mean ± std.

Checkpoint của lần chạy `--final` nằm ở `checkpoints/` (`mf_s*.pth`,
`lightgcn_s*.pth`), dùng cho app demo.
