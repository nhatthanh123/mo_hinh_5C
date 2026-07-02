# mo_hinh_5C
# 💳 Ứng dụng Dự báo Rủi ro Tín dụng Khách hàng (Mô hình 5C)

Ứng dụng Streamlit tái hiện quy trình trong notebook `Untitled3.ipynb`: huấn luyện
mô hình **Logistic Regression** để dự báo xác suất khách hàng có rủi ro tín dụng
(`PD`) dựa trên 24 biến khảo sát thuộc khung 5C:

- **TC** (TC1–TC5): Tư cách (Character)
- **NL** (NL1–NL4): Năng lực (Capacity)
- **DK** (DK1–DK5): Điều kiện (Conditions)
- **V** (V1–V6): Vốn (Capital)
- **TS** (TS1–TS4): Tài sản đảm bảo (Collateral)

Biến mục tiêu `PD`: `0` = không có rủi ro, `1` = có rủi ro.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc dữ liệu đầu vào

File CSV cần chứa tối thiểu các cột sau (giống `5c.csv` mẫu):

| Cột | Ý nghĩa |
|---|---|
| `TC1`...`TC5`, `NL1`...`NL4`, `DK1`...`DK5`, `V1`...`V6`, `TS1`...`TS4` | Điểm đánh giá (thang 1–5) cho từng tiêu chí 5C |
| `PD` | Nhãn rủi ro tín dụng (0/1) — dùng để huấn luyện & kiểm định |

Các cột khác trong file mẫu (`Dấu thời gian`, `NN`) không được notebook gốc đưa
vào mô hình nên ứng dụng cũng không sử dụng, để đảm bảo tái hiện đúng pipeline.

## Mô tả các tab

1. **📋 Tổng quan dữ liệu** — kích thước dữ liệu, xem nhanh bảng thô, thống kê mô
   tả cho các biến của mô hình (X và PD).
2. **📊 Trực quan hóa dữ liệu** — biểu đồ phân phối biến mục tiêu `PD`, cho phép
   chọn thêm tối đa 3 biến đầu vào để trực quan hóa cùng lúc (bố cục 2×2).
3. **🎯 Kết quả huấn luyện & kiểm định** — Accuracy, Precision, Recall, F1-score,
   ROC-AUC, ma trận nhầm lẫn, classification report và đường cong ROC (chỉ hiển
   thị sau khi bấm **Huấn luyện mô hình** ở thanh bên).
4. **🔮 Sử dụng mô hình** — dự báo cho khách hàng mới bằng cách nhập trực tiếp
   điểm 5C, hoặc tải lên file CSV theo đúng cấu trúc `X_test` để dự báo hàng loạt
   và tải kết quả về (CSV, mã hoá UTF-8-SIG).

## Ghi chú kỹ thuật

- Notebook gốc **không sử dụng scaler/encoder** nào trước khi huấn luyện
  `LogisticRegression`, nên ứng dụng cũng giữ nguyên dữ liệu thô khi huấn luyện
  và dự báo để đảm bảo tái hiện đúng kết quả (Accuracy ≈ 86.7% với
  `test_size=0.2`, `random_state=23` như notebook gốc).
- Các siêu tham số của `LogisticRegression` (solver, penalty, C, max_iter)
  không được đặt tường minh trong notebook (dùng giá trị mặc định của
  scikit-learn), nên ứng dụng dùng đúng các giá trị mặc định này và cho phép
  người dùng tinh chỉnh thêm trong mục "Tham số nâng cao" ở thanh bên.
- Mô hình chỉ được huấn luyện **một lần** khi bấm nút ở sidebar, kết quả (mô
  hình đã fit, bảng kết quả kiểm định) được lưu trong `st.session_state` để các
  tab dùng lại mà không cần train lại khi chuyển tab.
- Khuyến nghị dùng **Streamlit ≥ 1.38** (mới hơn cho hỗ trợ ổn định các thành
  phần layout hiện đại như `st.container(height=...)`).
