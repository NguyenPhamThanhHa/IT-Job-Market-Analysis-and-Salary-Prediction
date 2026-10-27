import os
import joblib
import pandas as pd
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự đoán mức lương IT Toàn cầu",
    layout="centered"
)

# Sử dụng CSS để căn giữa toàn bộ tiêu đề chính (h1)
st.markdown("""
    <style>
    h1 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Đường dẫn thư mục models
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
models_dir = os.path.join(root_dir, "models")

# Load model và các encoder (Sử dụng cache để tối ưu tốc độ)
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(models_dir, "salary_model.pkl"))
    job_encoder = joblib.load(os.path.join(models_dir, "job_encoder.pkl"))
    location_encoder = joblib.load(os.path.join(models_dir, "location_encoder.pkl"))
    return model, job_encoder, location_encoder

try:
    model, job_encoder, location_encoder = load_artifacts()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}. Hãy đảm bảo bạn đã chạy file train_model.py trước!")
    st.stop()

# Tiêu đề chính (đã được căn giữa bằng CSS, đã xóa dòng mô tả đồ án)
st.title("Hệ Thống Dự Đoán Mức Lương Lập Trình Viên Toàn Cầu")

st.divider()

# Form nhập liệu
with st.form("prediction_form"):
    st.subheader("Nhập thông tin truy vấn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        work_year = st.number_input("Năm làm việc", min_value=2020, max_value=2030, value=2026, step=1)
        
    with col2:
        job_title = st.selectbox("Vị trí công việc", options=list(job_encoder.classes_))
        
    # Danh sách quốc gia (Mặc định đưa VN lên vị trí dễ thấy nếu có)
    all_locations = list(location_encoder.classes_)
    default_idx = all_locations.index("VN") if "VN" in all_locations else 0
    
    company_location = st.selectbox("Mã quốc gia làm việc (ví dụ: VN, US, GB...)", options=all_locations, index=default_idx)

    st.markdown("<br>", unsafe_allow_html=True)

    # Canh giữa nút bấm
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col2:
        submitted = st.form_submit_button("Dự đoán mức lương", use_container_width=True)

# Xử lý kết quả khi bấm nút
if submitted:
    try:
        # Chuyển đổi nhãn chữ sang số bằng Encoder
        encoded_job = job_encoder.transform([job_title])[0]
        encoded_location = location_encoder.transform([company_location])[0]

        # Tạo DataFrame đầu vào
        X_input = pd.DataFrame(
            [[work_year, encoded_job, encoded_location]],
            columns=["work_year", "job_title_encoded", "company_location_encoded"]
        )

        # Dự đoán mức lương (USD)
        predicted_salary_usd = model.predict(X_input)[0]

        # Hiển thị kết quả trực quan
        st.divider()
        st.subheader("Kết Quả Dự Đoán")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Năm làm việc", value=work_year)
        c2.metric(label="Vị trí", value=job_title)
        c3.metric(label="Quốc gia", value=company_location)

        st.success(f"Mức lương dự kiến: **${predicted_salary_usd:,.2f} USD / năm**")

    except Exception as ex:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {ex}")