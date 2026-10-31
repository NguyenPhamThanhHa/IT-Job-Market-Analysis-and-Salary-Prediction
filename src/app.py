import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Cấu hình trang Streamlit ---
st.set_page_config(page_title="Ứng Dụng Dự Đoán Lương Ngành CNTT", layout="wide")

# --- TÙY CHỈNH CSS: Thu hẹp thanh bên (Sidebar) vừa phải và làm đẹp giao diện ---
st.markdown(
  """
    <style>
    [data-testid="stSidebar"] {
        min-width: 240px; 
        max-width: 280px;
    }
    
    /* Chỉnh size chữ Tiêu đề lớn (h1) */
    .centered-title {
        text-align: center;
        font-size: 36px; /* <--- SỬA SỐ NÀY ĐỂ TĂNG/GIẢM CỠ CHỮ TIÊU ĐỀ (VD: 30px, 40px) */
        font-weight: bold;
    }
    
    /* Chỉnh size chữ Đoạn mô tả bên dưới */
    .centered-text {
        text-align: center;
        color: #555555;
        font-size: 18px; /* <--- SỬA SỐ NÀY ĐỂ TĂNG/GIẢM CỠ CHỮ MÔ TẢ (VD: 16px, 20px) */
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. Tải và huấn luyện mô hình ML ---
@st.cache_resource
def load_and_train_model():
    data = {
        'work_year': [2020, 2021, 2022, 2023, 2024, 2022, 2023, 2024],
        'experience_level': ['Fresher', 'Junior', 'Middle', 'Senior', 'Senior', 'Fresher', 'Junior', 'Senior'],
        'job_title': ['AI Engineer', 'Data Scientist', 'Data Engineer', 'AI Engineer', 'Machine Learning Engineer', 'AI Engineer', 'Data Analyst', 'Data Scientist'],
        'company_location': ['France', 'United States', 'Germany', 'United Kingdom', 'United States', 'France', 'Canada', 'France'],
        'salary_in_usd': [45000, 95000, 110000, 150000, 180000, 48000, 75000, 140000]
    }
    df = pd.DataFrame(data)

    categorical_cols = ['experience_level', 'job_title', 'company_location']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, encoders

model, encoders = load_and_train_model()

# --- 2. Thanh bên (Sidebar) - Tùy chọn Ngôn ngữ ---
with st.sidebar:
    language = st.selectbox(
        "Ngôn ngữ / Language",
        ["Tiếng Việt", "English"]
    )

# --- 3. Từ điển đa ngôn ngữ (Localization Dictionary) ---
texts = {
    "Tiếng Việt": {
        "title": "Ứng Dụng Dự Đoán Lương Ngành Công Nghệ Thông Tin",
        "desc": "Nhập thông tin bên dưới để dự đoán mức lương phù hợp dựa trên mô hình học máy (Machine Learning).",
        "year": "Năm làm việc",
        "exp": "Cấp độ kinh nghiệm",
        "job": "Vị trí công việc",
        "loc": "Quốc gia làm việc",
        "btn": "Dự Đoán Lương",
        "success": "Mức lương dự đoán: **${predicted_salary:,.2f} USD / năm**",
        "error": "Đã xảy ra lỗi: "
    },
    "English": {
        "title": "IT Industry Salary Prediction Application",
        "desc": "Enter the information below to predict the appropriate salary based on a machine learning model.",
        "year": "Work Year",
        "exp": "Experience Level",
        "job": "Job Title",
        "loc": "Company Location",
        "btn": "Predict Salary",
        "success": "Predicted Salary: **${predicted_salary:,.2f} USD / year**",
        "error": "An error occurred: "
    }
}

t = texts[language]

# --- 4. Giao diện chính (Căn giữa nội dung bằng Columns) ---
# Tạo bố cục 3 cột: Cột trái và phải để trống làm khoảng đệm, cột giữa chứa nội dung chính
col_left, col_center, col_right = st.columns([1, 4, 1])

with col_center:
    st.title(t["title"])
    st.write(t["desc"])

    st.markdown("<br>", unsafe_allow_html=True)

    work_year = st.selectbox(
        t["year"],
        [2020, 2021, 2022, 2023, 2024, 2025, 2026],
        index=2
    )

    exp_options = ["Fresher", "Junior", "Middle", "Senior", "Executive"]
    experience_level = st.selectbox(
        t["exp"],
        exp_options,
        index=0
    )

    job_options = ["AI Engineer", "Data Scientist", "Data Engineer", "Data Analyst", "Machine Learning Engineer", "Software Engineer"]
    job_title = st.selectbox(
        t["job"],
        job_options,
        index=0
    )

    country_options = ["France", "United States", "United Kingdom", "Germany", "Canada", "Japan", "India"]
    company_location = st.selectbox(
        t["loc"],
        country_options,
        index=0
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(t["btn"], use_container_width=True):
        try:
            exp_val = encoders['experience_level'].transform([experience_level])[0] if experience_level in encoders['experience_level'].classes_ else 0
            job_val = encoders['job_title'].transform([job_title])[0] if job_title in encoders['job_title'].classes_ else 0
            loc_val = encoders['company_location'].transform([company_location])[0] if company_location in encoders['company_location'].classes_ else 0

            input_df = pd.DataFrame({
                'work_year': [work_year],
                'experience_level': [exp_val],
                'job_title': [job_val],
                'company_location': [loc_val]
            })

            predicted_salary = model.predict(input_df)[0]

            st.success(t["success"].format(predicted_salary=predicted_salary))
        except Exception as e:
            st.error(f"{t['error']}{e}")