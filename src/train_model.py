import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Đọc dữ liệu từ file (Thay 'salary_data.csv' hoặc đường dẫn file của bạn)
# Nếu file của bạn phân tách bằng dấu phẩy (CSV) hoặc tab (TXT), điều chỉnh hàm read_csv/read_table tương ứng:
try:
    df = pd.read_csv('salary_data.csv') # Hoặc pd.read_csv('salary_data.txt', sep='\t')
except FileNotFoundError:
    # Dữ liệu giả lập chuẩn mô phỏng theo cấu trúc file Kaggle (bao gồm các quốc gia phổ biến)
    data = {
        'work_year': [2023, 2023, 2024, 2024, 2023, 2024, 2023],
        'experience_level': ['SE', 'MI', 'EN', 'SE', 'EX', 'MI', 'EN'],
        'employment_type': ['FT', 'FT', 'FT', 'FT', 'FT', 'FT', 'FT'],
        'job_title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'ML Engineer', 'Data Scientist', 'Data Analyst', 'Data Engineer'],
        'salary_in_usd': [140000, 95000, 60000, 160000, 200000, 85000, 50000],
        'company_location': ['US', 'GB', 'CA', 'US', 'DE', 'IN', 'GB'],
        'company_size': ['M', 'L', 'S', 'L', 'M', 'L', 'S']
    }
    df = pd.DataFrame(data)

print("--- CÁC QUỐC GIA CÓ TRONG TẬP DỮ LIỆU ---")
countries = df['company_location'].unique()
print(countries)

# 2. Tiền xử lý dữ liệu & Mã hóa các biến phân loại (Categorical variables)
categorical_cols = ['experience_level', 'employment_type', 'job_title', 'company_location', 'company_size']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # Lưu lại encoder để sau này nếu cần giải mã ngược

# 3. Chia tập dữ liệu thành Train (80%) và Test (20%)
X = df.drop(columns=['salary_in_usd'])
y = df['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện mô hình Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH ---")
print(f"Sai số tuyệt đối trung bình (MAE): ${mae:,.2f}")
print(f"Căn bậc hai sai số bình phương (RMSE): ${rmse:,.2f}")
print(f"Hệ số xác định (R² Score): {r2:.4f}")

# 6. Ví dụ thực hiện dự đoán lương cho một mẫu mới theo quốc gia cụ thể
# Giả sử dự đoán cho một vị trí Senior Data Scientist tại Mỹ (US) và Anh (GB)
print("\n--- KẾT QUẢ DỰ ĐOÁN MẪU ---")
sample_data = pd.DataFrame({
    'work_year': [2024, 2024],
    'experience_level': [encoders['experience_level'].transform(['SE'])[0], encoders['experience_level'].transform(['SE'])[0]],
    'employment_type': [encoders['employment_type'].transform(['FT'])[0], encoders['employment_type'].transform(['FT'])[0]],
    'job_title': [encoders['job_title'].transform(['Data Scientist'])[0], encoders['job_title'].transform(['Data Scientist'])[0]],
    'company_location': [encoders['company_location'].transform(['US'])[0], encoders['company_location'].transform(['GB'])[0]],
    'company_size': [encoders['company_size'].transform(['M'])[0], encoders['company_size'].transform(['M'])[0]]
})

predicted_salaries = model.predict(sample_data)
print(f"Dự đoán lương tại Mỹ (US): ${predicted_salaries[0]:,.2f}")
print(f"Dự đoán lương tại Anh (GB): ${predicted_salaries[1]:,.2f}")