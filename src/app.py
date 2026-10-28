import os
import joblib
import pandas as pd
import streamlit as st

# =========================
# 1. CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Dự đoán mức lương IT toàn cầu",
    layout="centered"
)

# =========================
# 2. CSS GIAO DIỆN
# =========================
st.markdown(
    """
    <style>
    h1 {
        text-align: center;
    }
    .stForm {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 3. XÁC ĐỊNH ĐƯỜNG DẪN
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))

possible_model_dirs = [
    os.path.join(current_dir, "models"),
    os.path.join(os.path.dirname(current_dir), "models")
]

models_dir = next(
    (
        path for path in possible_model_dirs
        if os.path.isdir(path)
    ),
    possible_model_dirs[0]
)

# =========================
# 4. TẢI MÔ HÌNH VÀ ENCODER
# =========================
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(models_dir, "salary_model.pkl")
    job_encoder_path = os.path.join(models_dir, "job_encoder.pkl")
    location_encoder_path = os.path.join(models_dir, "location_encoder.pkl")

    required_files = [model_path, job_encoder_path, location_encoder_path]

    for file_path in required_files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    model = joblib.load(model_path)
    job_encoder = joblib.load(job_encoder_path)
    location_encoder = joblib.load(location_encoder_path)

    return model, job_encoder, location_encoder

try:
    model, job_encoder, location_encoder = load_artifacts()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.info(
        "Hãy kiểm tra thư mục models và bảo đảm "
        "đã chạy train_model.py để tạo các file cần thiết."
    )
    st.stop()

# =========================
# 5. TIÊU ĐỀ CHƯƠNG TRÌNH
# =========================
st.title("Hệ Thống Dự Đoán Mức Lương Lập Trình Viên Toàn Cầu")
st.write(
    "Nhập thông tin công việc (Front End, Back End, Mobile, AI Engineer) "
    "để dự đoán mức lương hằng năm bằng mô hình Machine Learning."
)

st.divider()

# =========================
# 6. CHUẨN BỊ DANH SÁCH LỰA CHỌN
# =========================
job_options = job_encoder.classes_.tolist()
location_options = location_encoder.classes_.tolist()

# Từ điển ánh xạ mã quốc gia sang tên đầy đủ (đã bổ sung Đức, Pháp, Trung Quốc...)
country_display_names = {
    "VN": "Việt Nam (VN)",
    "US": "Mỹ / Hoa Kỳ (US)",
    "DE": "Đức (DE)",
    "FR": "Pháp (FR)",
    "JP": "Nhật Bản (JP)",
    "KR": "Hàn Quốc (KR)",
    "SG": "Singapore (SG)",
    "CN": "Trung Quốc (CN)",
    "GB": "Vương quốc Anh (GB)"
}

display_location_options = [country_display_names.get(loc, loc) for loc in location_options]

# Ưu tiên Việt Nam nếu có trong dữ liệu
default_location_index = 0
for idx, loc in enumerate(location_options):
    if loc == "VN":
        default_location_index = idx
        break

# =========================
# 7. FORM NHẬP THÔNG TIN
# =========================
with st.form("prediction_form"):
    st.subheader("Nhập thông tin truy vấn")

    col1, col2 = st.columns(2)

    with col1:
        work_year = st.number_input(
            "Năm làm việc",
            min_value=2022,
            max_value=2026,
            value=2026,
            step=1
        )

    with col2:
        job_title = st.selectbox(
            "Vị trí công việc",
            options=job_options
        )

    selected_display_location = st.selectbox(
        "Quốc gia làm việc",
        options=display_location_options,
        index=default_location_index,
        help="Chọn quốc gia bạn muốn dự đoán mức lương."
    )

    # Lấy lại mã code gốc từ tên hiển thị
    selected_idx = display_location_options.index(selected_display_location)
    company_location = location_options[selected_idx]

    st.caption("Mức lương dự đoán được tính bằng USD mỗi năm (Dữ liệu từ 2022 - 2026).")
    st.write("")

    col_left, col_button, col_right = st.columns([1, 2, 1])

    with col_button:
        submitted = st.form_submit_button(
            "Dự đoán mức lương",
            use_container_width=True
        )

# =========================
# 8. XỬ LÝ DỰ ĐOÁN
# =========================
if submitted:
    try:
        # Mã hóa vị trí công việc và quốc gia
        encoded_job = job_encoder.transform([job_title])[0]
        encoded_location = location_encoder.transform([company_location])[0]

        # Tạo dữ liệu đầu vào cho mô hình
        X_input = pd.DataFrame(
            [[work_year, encoded_job, encoded_location]],
            columns=[
                "work_year",
                "job_title_encoded",
                "company_location_encoded"
            ]
        )

        # Dự đoán mức lương
        predicted_salary_usd = float(model.predict(X_input)[0])

        if not pd.notna(predicted_salary_usd):
            raise ValueError("Mô hình trả về kết quả không hợp lệ.")

        # =========================
        # 9. HIỂN THỊ KẾT QUẢ
        # =========================
        st.divider()
        st.subheader("Kết quả dự đoán")

        c1, c2, c3 = st.columns(3)

        c1.metric(label="Năm làm việc", value=str(work_year))
        c2.metric(label="Vị trí công việc", value=job_title)
        c3.metric(label="Quốc gia", value=company_location)

        if predicted_salary_usd < 0:
            st.warning(
                "Mức lương dự đoán âm. "
                "Hãy kiểm tra dữ liệu và mô hình huấn luyện."
            )
        else:
            st.success(
                f"Mức lương dự kiến: "
                f"**${predicted_salary_usd:,.2f} USD/năm**"
            )
            st.caption(
                "Kết quả chỉ mang tính tham khảo, "
                "phụ thuộc vào dữ liệu và mô hình đã huấn luyện."
            )

    except Exception as ex:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {ex}")