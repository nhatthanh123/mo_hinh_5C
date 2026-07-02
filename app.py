import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import train_test_split

# =========================================================
# 1) CẤU HÌNH TRANG (LỆNH STREAMLIT ĐẦU TIÊN)
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Dự báo Rủi ro Tín dụng Khách hàng (5C)",
    page_icon="💳",
)

# =========================================================
# 2) HẰNG SỐ & HÀM DÙNG CHUNG
# =========================================================

# Tập biến đầu vào X trích xuất từ notebook (bước 2)
FEATURE_COLS = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
TARGET_COL = "PD"

# Nhóm biến theo khung 5C (Character, Capacity, Conditions, Capital,
# Collateral) — chỉ dùng để chú thích giao diện, không ảnh hưởng mô hình.
GROUP_LABELS = {
    "TC": "Tư cách (Character)",
    "NL": "Năng lực (Capacity)",
    "DK": "Điều kiện (Conditions)",
    "V": "Vốn (Capital)",
    "TS": "Tài sản đảm bảo (Collateral)",
}


def group_of(col: str) -> str:
    for prefix, label in GROUP_LABELS.items():
        if col.startswith(prefix):
            return label
    return col


@st.cache_data(show_spinner="Đang nạp dữ liệu...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu từ bytes (để hashable với st.cache_data)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def validate_columns(df: pd.DataFrame, required_cols: list) -> list:
    """Trả về danh sách cột còn thiếu."""
    return [c for c in required_cols if c not in df.columns]


# =========================================================
# 3) SIDEBAR — VÙNG CẤU HÌNH (THÀNH PHẦN 1)
# =========================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu khảo sát khách hàng (.csv)",
        type=["csv"],
        help="File cần chứa các cột biến đầu vào (TC1..TS4) và cột nhãn PD.",
    )

    # Chỉ có DUY NHẤT một mô hình trong notebook (LogisticRegression)
    # → không hiển thị lựa chọn mô hình.
    st.subheader("Tham số mô hình AI")

    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test size)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm định mô hình. Notebook gốc dùng 0.2 (20%).",
    )
    random_state = st.number_input(
        "random_state (chia tập train/test)",
        min_value=0,
        max_value=9999,
        value=23,
        step=1,
        help="Giá trị notebook gốc đã dùng để đảm bảo kết quả tái lập được.",
    )

    with st.expander("Tham số nâng cao (Logistic Regression)"):
        solver = st.selectbox(
            "solver",
            options=["lbfgs", "liblinear", "newton-cg", "saga"],
            index=0,
            help="Thuật toán tối ưu. Notebook gốc dùng mặc định của sklearn (lbfgs).",
        )
        # Danh sách penalty hợp lệ theo từng solver (ràng buộc của sklearn)
        penalty_options = {
            "lbfgs": ["l2", None],
            "newton-cg": ["l2", None],
            "liblinear": ["l1", "l2"],
            "saga": ["l1", "l2", "elasticnet", None],
        }
        penalty = st.selectbox(
            "penalty",
            options=penalty_options[solver],
            index=0,
            help="Kiểu điều chuẩn (regularization). Mặc định notebook: l2.",
        )
        C = st.slider(
            "C (nghịch đảo cường độ điều chuẩn)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Giá trị mặc định của sklearn là 1.0 — notebook không thay đổi.",
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=50,
            max_value=2000,
            value=100,
            step=50,
            help="Số vòng lặp tối đa để hội tụ. Mặc định sklearn: 100.",
        )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
    )

# =========================================================
# 4) HEADER — VÙNG ĐỊNH HƯỚNG (THÀNH PHẦN 2)
# =========================================================
st.title("💳 Ứng dụng Dự báo Rủi ro Tín dụng Khách hàng (Mô hình 5C)")
st.caption(
    "Ứng dụng tái hiện quy trình huấn luyện Logistic Regression từ notebook: "
    "dự báo khả năng khách hàng có rủi ro tín dụng (PD) dựa trên 24 biến khảo sát "
    "thuộc 5 nhóm tiêu chí 5C (Tư cách, Năng lực, Điều kiện, Vốn, Tài sản đảm bảo)."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file dữ liệu (.csv) ở thanh bên để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()
df = load_data(file_bytes)

missing_cols = validate_columns(df, FEATURE_COLS + [TARGET_COL])
if missing_cols:
    st.error(
        "❌ File dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_cols)
    )
    st.stop()

if df.empty:
    st.error("❌ File dữ liệu rỗng. Vui lòng tải lên file khác.")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(
    f"Dữ liệu gồm **{df.shape[0]}** dòng và **{df.shape[1]}** cột. "
    f"Biến mục tiêu: **{TARGET_COL}** (0 = không rủi ro, 1 = có rủi ro)."
)
st.divider()

# =========================================================
# 5) KHỐI TRAIN — CHẠY KHI BẤM NÚT Ở SIDEBAR
# =========================================================
if train_clicked:
    try:
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=C,
            max_iter=int(max_iter),
        )
        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        proba_test = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        results = {
            "y_test": y_test.reset_index(drop=True),
            "yhat_test": pd.Series(yhat_test),
            "proba_test": pd.Series(proba_test) if proba_test is not None else None,
        }

        st.session_state["model"] = model
        st.session_state["preprocessor"] = None  # notebook không dùng scaler/encoder
        st.session_state["results"] = results
        st.session_state["feature_cols"] = FEATURE_COLS
        st.session_state["trained"] = True
        st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở các tab bên dưới.")
    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra khi huấn luyện mô hình: {e}")

# =========================================================
# 6) CÁC TAB CHỨA THÀNH PHẦN 3, 4, 5, 6
# =========================================================
tab_overview, tab_viz, tab_results, tab_predict = st.tabs(
    [
        "📋 Tổng quan dữ liệu",
        "📊 Trực quan hóa dữ liệu",
        "🎯 Kết quả huấn luyện & kiểm định",
        "🔮 Sử dụng mô hình",
    ]
)

# ---------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ---------------------------------------------------------
with tab_overview:
    file_size_mb = len(file_bytes) / (1024 * 1024)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", df.shape[0])
    c2.metric("Số cột", df.shape[1])
    c3.metric("Dung lượng file", f"{file_size_mb:.2f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả (biến của mô hình)")
    st.caption("Chỉ hiển thị các biến đầu vào (X) và biến mục tiêu (y) dùng trong mô hình.")
    st.dataframe(
        df[FEATURE_COLS + [TARGET_COL]].describe(),
        use_container_width=True,
    )

# ---------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ---------------------------------------------------------
with tab_viz:
    st.caption(
        "Biểu đồ mục tiêu (PD) hiển thị trước, sau đó là các biến đầu vào — "
        "chọn thêm tối đa 3 biến để xem cùng lúc (tổng cộng 4 biểu đồ)."
    )

    default_inputs = [c for c in ["TC1", "NL1", "DK1"] if c in FEATURE_COLS][:3]
    selected_inputs = st.multiselect(
        "Chọn biến đầu vào để trực quan hóa (tối đa 3)",
        options=FEATURE_COLS,
        default=default_inputs,
        max_selections=3,
        help="Vì có nhiều hơn 4 biến, hãy chọn các biến bạn muốn xem cùng biến mục tiêu.",
    )

    def make_chart(col: str, is_target: bool = False):
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            fig = px.histogram(df, x=col, title=f"Phân phối {col}")
        else:
            vc = series.value_counts().sort_index().reset_index()
            vc.columns = [col, "Số lượng"]
            if is_target:
                vc[col] = vc[col].map({0: "Không rủi ro (0)", 1: "Có rủi ro (1)"})
            fig = px.bar(vc, x=col, y="Số lượng", title=f"Phân phối {col} — {group_of(col)}" if not is_target else "Phân phối biến mục tiêu (PD)")
        fig.update_layout(height=350)
        return fig

    charts = [(TARGET_COL, True)] + [(c, False) for c in selected_inputs]

    rows = [charts[i:i + 2] for i in range(0, len(charts), 2)]
    for row in rows:
        cols = st.columns(2)
        for (col_name, is_target), slot in zip(row, cols):
            with slot:
                st.plotly_chart(make_chart(col_name, is_target), use_container_width=True)

# ---------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ---------------------------------------------------------
with tab_results:
    if not st.session_state.get("trained"):
        st.info("ℹ️ Vui lòng bấm nút **Huấn luyện mô hình** ở thanh bên để xem kết quả.")
        st.stop()

    results = st.session_state["results"]
    y_test = results["y_test"]
    yhat_test = results["yhat_test"]
    proba_test = results["proba_test"]

    acc = accuracy_score(y_test, yhat_test)
    prec = precision_score(y_test, yhat_test, zero_division=0)
    rec = recall_score(y_test, yhat_test, zero_division=0)
    f1 = f1_score(y_test, yhat_test, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Precision", f"{prec*100:.1f}%")
    m3.metric("Recall", f"{rec*100:.1f}%")
    m4.metric("F1-score", f"{f1*100:.1f}%")

    if proba_test is not None:
        auc = roc_auc_score(y_test, proba_test)
        st.metric("ROC-AUC", f"{auc:.3f}")

    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
    cm = confusion_matrix(y_test, yhat_test)
    cm_df = pd.DataFrame(
        cm,
        index=["Thực tế: Không rủi ro (0)", "Thực tế: Có rủi ro (1)"],
        columns=["Dự báo: Không rủi ro (0)", "Dự báo: Có rủi ro (1)"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification report")
    report = classification_report(y_test, yhat_test, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    if proba_test is not None:
        st.subheader("Đường cong ROC")
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        roc_fig = px.area(
            x=fpr, y=tpr,
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            title=f"ROC Curve (AUC = {auc:.3f})",
        )
        roc_fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        roc_fig.update_layout(height=400)
        st.plotly_chart(roc_fig, use_container_width=True)

# ---------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ---------------------------------------------------------
with tab_predict:
    if not st.session_state.get("trained"):
        st.info("ℹ️ Vui lòng bấm nút **Huấn luyện mô hình** ở thanh bên trước khi dự báo.")
        st.stop()

    model = st.session_state["model"]
    feature_cols = st.session_state["feature_cols"]

    mode = st.radio(
        "Chọn chế độ sử dụng",
        options=["Nhập trực tiếp", "Tải file dữ liệu mới"],
        horizontal=True,
    )

    if mode == "Nhập trực tiếp":
        st.caption("Nhập điểm đánh giá (1–5) cho từng tiêu chí 5C của khách hàng mới.")
        with st.form("predict_form"):
            input_values = {}
            cols_widgets = st.columns(4)
            for i, col in enumerate(feature_cols):
                col_data = df[col]
                with cols_widgets[i % 4]:
                    input_values[col] = st.number_input(
                        f"{col} ({group_of(col)})",
                        min_value=float(col_data.min()),
                        max_value=float(col_data.max()),
                        value=float(col_data.median()),
                        step=1.0,
                        help=f"Khoảng giá trị trong dữ liệu: {int(col_data.min())}–{int(col_data.max())}",
                    )
            submitted = st.form_submit_button("Dự báo", type="primary", use_container_width=True)

        if submitted:
            X_new = pd.DataFrame([input_values])[feature_cols]
            pred = model.predict(X_new)[0]
            if pred == 0:
                st.success(f"✅ Kết quả dự báo: **Khách hàng không có rủi ro tín dụng** (PD = {pred})")
            else:
                st.warning(f"⚠️ Kết quả dự báo: **Khách hàng có rủi ro tín dụng** (PD = {pred})")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]
                p1, p2 = st.columns(2)
                p1.metric("Xác suất không rủi ro", f"{proba[0]*100:.2f}%")
                p2.metric("Xác suất có rủi ro", f"{proba[1]*100:.2f}%")

    else:
        st.caption(
            "Tải lên file CSV chứa đúng các cột biến đầu vào (giống cấu trúc X_test) để dự báo hàng loạt."
        )
        batch_file = st.file_uploader(
            "Tải file dữ liệu khách hàng mới (.csv)",
            type=["csv"],
            key="batch_predict_uploader",
        )
        if batch_file is not None:
            try:
                new_df = pd.read_csv(io.BytesIO(batch_file.getvalue()))
                missing = validate_columns(new_df, feature_cols)
                if missing:
                    st.error("❌ File thiếu các cột bắt buộc: " + ", ".join(missing))
                else:
                    X_batch = new_df[feature_cols]
                    preds = model.predict(X_batch)
                    out_df = new_df.copy()
                    out_df["Dự báo PD"] = preds
                    if hasattr(model, "predict_proba"):
                        proba_batch = model.predict_proba(X_batch)
                        out_df["Xác suất không rủi ro (%)"] = (proba_batch[:, 0] * 100).round(2)
                        out_df["Xác suất có rủi ro (%)"] = (proba_batch[:, 1] * 100).round(2)

                    st.subheader("Kết quả dự báo")
                    with st.container(height=350):
                        st.dataframe(out_df, use_container_width=True)

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Tải kết quả dự báo (CSV)",
                        data=csv_bytes,
                        file_name="ket_qua_du_bao_PD.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra khi xử lý file: {e}")
