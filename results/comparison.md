# Kết quả thực nghiệm

Metric: Recall@20 / NDCG@20, full ranking, đã che item user đã biết.
HPO chọn cấu hình theo VALIDATION; TEST chỉ chạy 1 lần với 3 seed.

## MF

- Số thí nghiệm HPO: 12
- Cấu hình tốt nhất theo validation: `C_mf_d256_lr0.001_reg0.001` (dim=256, layers=?, lr=0.001, reg=0.001)

| Seed | Recall@20 | NDCG@20 |
|---|---|---|
| 0 | 0.1198 | 0.0771 |
| 1 | 0.1190 | 0.0760 |
| 2 | 0.1195 | 0.0762 |
| **TB ± SD** | **0.1194 ± 0.0004** | **0.0764 ± 0.0006** |

## LightGCN

- Số thí nghiệm HPO: 11
- Cấu hình tốt nhất theo validation: `A_lgcn_L2_d128` (dim=128, layers=2, lr=0.001, reg=0.0001)

| Seed | Recall@20 | NDCG@20 |
|---|---|---|
| 0 | 0.1232 | 0.0781 |
| 1 | 0.1240 | 0.0793 |
| 2 | 0.1206 | 0.0770 |
| **TB ± SD** | **0.1226 ± 0.0018** | **0.0781 ± 0.0012** |

## Baseline tham chiếu

| Model | Split | Recall@20 | NDCG@20 |
|---|---|---|---|
| MostPop | val | 0.0914 | 0.0638 |
| MostPop | test | 0.0778 | 0.0557 |

