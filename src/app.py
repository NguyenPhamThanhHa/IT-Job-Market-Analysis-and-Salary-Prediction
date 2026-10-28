import os
import pickle
import streamlit as st
import pandas as pd

# Thiết lập đường dẫn thư mục chứa model và encoders
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'models')

# Load model và các encoders
@st.cache_resource
def load_artifacts():
    try:
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        encoders = {}
        categorical_cols = ['experience_level', 'job_title', 'company_location']
        for col in categorical_cols:
            with open(os.path.join(model_dir, f"{col}_encoder.pkl"), 'rb') as f:
                encoders[col] = pickle.load(f)
        return model, encoders
    except Exception as e:
        return None, None

model, encoders = load_artifacts()

# Từ điển ngôn ngữ (Đã loại bỏ các icon)
translations = {
    "Tiếng Việt": {
        "title": "Ứng Dụng Dự Đoán Lương Ngành Công Nghệ Thông Tin",
        "desc": "Nhập thông tin bên dưới để dự đoán mức lương phù hợp dựa trên mô hình học máy (Machine Learning).",
        "year_label": "Năm làm việc",
        "exp_label": "Cấp độ kinh nghiệm",
        "job_label": "Vị trí công việc",
        "loc_label": "Quốc gia làm việc",
        "btn_text": "Dự Đoán Lương",
        "result": "Mức lương dự kiến:",
        "error_load": "Không tìm thấy mô hình hoặc các tệp mã hóa! Hãy chạy tệp huấn luyện trước.",
        "error_pred": "Đã xảy ra lỗi trong quá trình dự đoán:"
    },
    "English": {
        "title": "IT Salary Prediction App",
        "desc": "Enter the details below to predict the estimated salary using our Machine Learning model.",
        "year_label": "Work Year",
        "exp_label": "Experience Level",
        "job_label": "Job Title",
        "loc_label": "Company Location",
        "btn_text": "Predict Salary",
        "result": "Estimated Salary:",
        "error_load": "Model or encoder files not found! Please run the training script first.",
        "error_pred": "An error occurred during prediction:"
    }
}

# Chọn ngôn ngữ ở thanh bên (Sidebar)
selected_lang = st.sidebar.selectbox("Ngôn ngữ / Language", ["Tiếng Việt", "English"])
t = translations[selected_lang]

# Giao diện chính
st.title(t["title"])
st.write(t["desc"])

if model is None or encoders is None:
    st.error(t["error_load"])
else:
    # Các ô lựa chọn lấy trực tiếp từ LabelEncoder đã train
    work_year = st.selectbox(t["year_label"], [2022, 2023, 2024, 2025, 2026])
    
    experience_level = st.selectbox(
        t["exp_label"], 
        encoders['experience_level'].classes_
    )
    
    job_title = st.selectbox(
        t["job_label"], 
        encoders['job_title'].classes_
    )
    
    company_location = st.selectbox(
        t["loc_label"], 
        encoders['company_location'].classes_
    )

    # Nút bấm dự đoán (Không còn icon)
    if st.button(t["btn_text"]):
        try:
            # Chuẩn bị dữ liệu đúng với tên cột lúc train mô hình
            input_data = pd.DataFrame({
                'work_year': [work_year],
                'experience_level_encoded': [encoders['experience_level'].transform([experience_level])[0]],
                'job_title_encoded': [encoders['job_title'].transform([job_title])[0]],
                'company_location_encoded': [encoders['company_location'].transform([company_location])[0]]
            })
            
            # Tiến hành dự đoán
            prediction = model.predict(input_data)[0]
            
            # Hiển thị kết quả
            st.success(f"{t['result']} **${prediction:,.2f} USD**")
        except Exception as e:
            st.error(f"{t['error_pred']} {e}")