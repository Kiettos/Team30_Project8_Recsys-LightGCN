# Phân tích lỗi (sinh tự động bởi src/error_analysis.py)

- Dữ liệu: 6034 user, 3125 item, test 57470 cặp (split temporal)
- Số seed dùng: 0 (trung bình các seed nếu >1)
- Metric: Recall@20 / NDCG@20, đã che train + val

## Tổng quan

| Model | Recall@20 | NDCG@20 | User không có hit | Tỉ lệ cặp test vào top-20 |
|---|---|---|---|---|
| MF | 0.1197 | 0.0770 | 3,373 (55.9%) | 7.50% |
| LightGCN | 0.1232 | 0.0781 | 3,407 (56.5%) | 7.62% |

## Recall theo mức hoạt động của user (chia 4 nhóm theo số tương tác train)

| Model | Nhóm | Số user | Khoảng tương tác | Recall@20 | NDCG@20 |
|---|---|---|---|---|---|
| MF | 1 | 1,518 | 3–21 | 0.2041 | 0.0965 |
| MF | 2 | 1,513 | 22–46 | 0.1333 | 0.0766 |
| MF | 3 | 1,495 | 47–99 | 0.0841 | 0.0645 |
| MF | 4 | 1,508 | 100–1133 | 0.0564 | 0.0702 |
| LightGCN | 1 | 1,518 | 3–21 | 0.2165 | 0.0978 |
| LightGCN | 2 | 1,513 | 22–46 | 0.1321 | 0.0774 |
| LightGCN | 3 | 1,495 | 47–99 | 0.0852 | 0.0643 |
| LightGCN | 4 | 1,508 | 100–1133 | 0.0578 | 0.0726 |

## Tỉ lệ CẶP test vào top-20, chia theo mức hoạt động của user

(Chỉ số công bằng hơn Recall trung bình: Recall mỗi user bị chia cho số item test của user đó,
nên user ít tương tác thường có Recall cao hơn một cách máy móc.)

| Model | Nhóm hoạt động | Tỉ lệ cặp test vào top-20 |
|---|---|---|
| MF | 1 | 12.67% |
| MF | 2 | 7.17% |
| MF | 3 | 5.58% |
| MF | 4 | 4.50% |
| LightGCN | 1 | 12.91% |
| LightGCN | 2 | 7.29% |
| LightGCN | 3 | 5.44% |
| LightGCN | 4 | 4.76% |

## Cặp test vào top-20, chia theo độ phổ biến của phim (nhóm 1 = phim hiếm nhất)

| Model | Nhóm | Độ phổ biến TB (số user trong train) | Tỉ lệ vào top-20 |
|---|---|---|---|
| MF | 1 | 53.9 | 0.51% |
| MF | 2 | 181.5 | 2.20% |
| MF | 3 | 393.3 | 5.93% |
| MF | 4 | 1013.5 | 21.43% |
| LightGCN | 1 | 53.9 | 0.65% |
| LightGCN | 2 | 181.5 | 2.56% |
| LightGCN | 3 | 393.3 | 6.26% |
| LightGCN | 4 | 1013.5 | 21.08% |

## Thiên lệch popularity & độ phủ

| Model | Điểm phổ biến TB của phim được gợi ý (0–1) | % gợi ý thuộc nhóm hiếm (bottom 25%) | Độ phủ catalog |
|---|---|---|---|
| MF | 0.935 | 0.0% | 53.2% |
| LightGCN | 0.930 | 0.0% | 54.0% |

Ghi chú: 'điểm phổ biến' = 1 - thứ_hạng_phổ_biến/(n_items-1), 1 là phim phổ biến nhất.
Độ phủ catalog = số phim khác nhau từng xuất hiện trong top-20 / tổng số phim.
