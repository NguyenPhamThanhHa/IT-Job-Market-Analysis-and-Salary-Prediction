import os
import joblib
import pandas as pd
import streamlit as st

# =========================
# 1. CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Global IT Salary Intelligence Dashboard",
    layout="wide"
)

# =========================
# 2. CUSTOM CSS (ĐỒNG BỘ IN HOA TIÊU ĐỀ SIDEBAR VÀ GIAO DIỆN)
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f1f5f9;
    }
    
    /* Mở rộng chiều rộng Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
        min-width: 420px !important;
    }

    /* Tiêu đề trong Sidebar: IN HOA TOÀN BỘ, TO VÀ ĐẬM */
    [data-testid="stSidebar"] .sidebar-title-container {
        text-align: center;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] .sidebar-title-container h2 {
        font-size: 1.3rem !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] .sidebar-title-container p {
        font-size: 0.95rem !important;
        color: #64748b !important;
        margin: 0;
    }

    [data-testid="stSidebar"] label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        white-space: normal !important;
        overflow: visible !important;
    }

    /* Header chính canh giữa */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .hero-header h1 {
        color: white !important;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Tiêu đề phần kết quả phân tích: VIẾT HOA TOÀN BỘ, TO VÀ ĐẬM */
    .section-title {
        text-align: center;
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Khung thẻ tùy chỉnh cân đối, chữ to rõ theo chiều ngang */
    .custom-metric-card {
        background-color: white;
        padding: 22px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .custom-metric-label {
        font-size: 0.95rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .custom-metric-value {
        font-size: 1.35rem !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
    }

    /* Card kết quả chính */
    .result-card {
        background: white;
        padding: 28px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-top: 15px;
    }

    /* Nút bấm dự đoán lớn và nổi bật */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 1rem;
        width: 100%;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 3. XÁC ĐỊNH ĐƯỜNG DẪN MODELS
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_model_dirs = [
    os.path.join(current_dir, "models"),
    os.path.join(os.path.dirname(current_dir), "models")
]
models_dir = next(
    (path for path in possible_model_dirs if os.path.isdir(path)),
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

    for file_path in [model_path, job_encoder_path, location_encoder_path]:
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
    st.stop()

# =========================
# 5. TIÊU ĐỀ TRANG CHÍNH
# =========================
st.markdown(
    """
    <div class="hero-header">
        <h1>Global IT Salary Intelligence</h1>
        <p>Hệ thống dự đoán và phân tích mức lương lập trình viên toàn cầu (2022 - 2026) bằng Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# 6. CHUẨN BỊ DỮ LIỆU LỰA CHỌN
# =========================
job_options = job_encoder.classes_.tolist()
location_options = location_encoder.classes_.tolist()

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

default_location_index = 0
for idx, loc in enumerate(location_options):
    if loc == "VN":
        default_location_index = idx
        break

# =========================
# 7. THANH ĐIỀU KHIỂN BÊN SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title-container">
            <h2>Thông tin tra cứu</h2>
            <p>Chọn tiêu chí công việc để hệ thống đánh giá.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.form("prediction_form"):
        work_year = st.selectbox(
            "Năm làm việc",
            options=[2022, 2023, 2024, 2025, 2026],
            index=4
        )

        job_title = st.selectbox(
            "Vị trí công việc",
            options=job_options
        )

        selected_display_location = st.selectbox(
            "Quốc gia làm việc",
            options=display_location_options,
            index=default_location_index
        )

        selected_idx = display_location_options.index(selected_display_location)
        company_location = location_options[selected_idx]

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Dự đoán mức lương")

# =========================
# 8. KHU VỰC HIỂN THỊ KẾT QUẢ
# =========================
if submitted:
    try:
        encoded_job = job_encoder.transform([job_title])[0]
        encoded_location = location_encoder.transform([company_location])[0]

        X_input = pd.DataFrame(
            [[work_year, encoded_job, encoded_location]],
            columns=["work_year", "job_title_encoded", "company_location_encoded"]
        )

        predicted_salary_usd = float(model.predict(X_input)[0])

        if not pd.notna(predicted_salary_usd) or predicted_salary_usd < 0:
            st.warning("Mô hình trả về kết quả không hợp lệ.")
        else:
            st.markdown('<div class="section-title">Kết Quả Phân Tích</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="custom-metric-card">
                        <div class="custom-metric-label">Năm đánh giá</div>
                        <div class="custom-metric-value">{work_year}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="custom-metric-card">
                        <div class="custom-metric-label">Vị trí chuyên môn</div>
                        <div class="custom-metric-value" title="{job_title}">{job_title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="custom-metric-card">
                        <div class="custom-metric-label">Thị trường</div>
                        <div class="custom-metric-value">{company_location}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="result-card">
                    <h3 style="color: #64748b; margin-bottom: 5px; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;">Ước tính thu nhập trung bình hàng năm</h3>
                    <h1 style="color: #2563eb; font-size: 2.3rem; font-weight: 700; margin: 0;">${predicted_salary_usd:,.2f} <span style="font-size: 1.1rem; color: #64748b; font-weight: 400;">USD / năm</span></h1>
                    <p style="color: #64748b; margin-top: 8px; font-size: 0.8rem;">Dựa trên mô hình Random Forest Regressor với dữ liệu thị trường quốc tế trong 5 năm gần đây.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as ex:
        st.error(f"Đã xảy ra lỗi: {ex}")
else:
    st.info("Sử dụng bảng điều khiển bên trái để thiết lập thông số và nhấn 'Dự đoán mức lương' để bắt đầu.")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-label">Thị trường hỗ trợ</div>
                <div class="custom-metric-value" style="font-size: 1.25rem;">9 Quốc gia lớn</div>
                <div style="color: #2563eb; font-weight: 500; font-size: 0.85rem; margin-top: 4px;">VN, Mỹ, EU, Châu Á</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_b:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-label">Ngành nghề cốt lõi</div>
                <div class="custom-metric-value" style="font-size: 1.25rem;">4 Nhóm chuyên sâu</div>
                <div style="color: #2563eb; font-weight: 500; font-size: 0.85rem; margin-top: 4px;">Front End, Back End, Mobile, AI</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_c:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-label">Dữ liệu cập nhật</div>
                <div class="custom-metric-value" style="font-size: 1.25rem;">5 Năm gần nhất</div>
                <div style="color: #2563eb; font-weight: 500; font-size: 0.85rem; margin-top: 4px;">2022 - 2026</div>
            </div>
            """,
            unsafe_allow_html=True
        )