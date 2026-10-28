import os
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Tạo thư mục models nếu chưa có
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# 1. Tải tập dữ liệu
data_path = os.path.join(current_dir, "salaries.csv")

try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"Không tìm thấy file dữ liệu: {e}")
    # Dữ liệu mẫu giả lập trong 5 năm gần đây (2022 - 2026) với đầy đủ các quốc gia yêu cầu
    df = pd.DataFrame({
        "work_year": [2022, 2023, 2024, 2025, 2026, 2023, 2024, 2025, 2026, 2024, 2025, 2026, 2024, 2025],
        "job_title": [
            "Front End Developer", "Back End Engineer", "Mobile Developer", 
            "AI Engineer", "Front End Developer", "Back End Engineer", 
            "AI Engineer", "Mobile Developer", "Front End Developer", "AI Engineer", "Back End Engineer", "Mobile Developer", "Front End Developer", "AI Engineer"
        ],
        "company_location": ["VN", "US", "DE", "FR", "JP", "KR", "SG", "CN", "GB", "US", "DE", "VN", "FR", "JP"],
        "salary_in_usd": [22000, 105000, 75000, 70000, 30000, 82000, 39000, 55000, 120000, 99000, 78000, 31000, 72000, 96000]
    })

print("Đang tiến hành lọc dữ liệu...")

# =====================================================================
# BỘ LỌC 1: Chỉ lấy 5 năm gần đây (2022 - 2026)
# =====================================================================
current_year = 2026
start_year = current_year - 4  # 2022 đến 2026
df = df[df['work_year'] >= start_year]

# =====================================================================
# BỘ LỌC 2: Chỉ giữ lại 4 nhóm công việc (Front End, Back End, Mobile, AI)
# =====================================================================
pattern = r'front|back|mobile|ai|machine learning'
df = df[df['job_title'].str.contains(pattern, case=False, na=False)]

# =====================================================================
# BỘ LỌC 3: Chỉ giữ lại danh sách các quốc gia theo yêu cầu
# =====================================================================
target_countries = ["VN", "US", "DE", "FR", "JP", "KR", "SG", "CN", "GB"]
df = df[df['company_location'].isin(target_countries)]

if df.empty:
    print("Cảnh báo: Dữ liệu sau khi lọc bị trống! Hãy kiểm tra lại file CSV hoặc bộ lọc.")
else:
    print(f"Đã lọc xong. Số lượng mẫu dữ liệu phù hợp: {len(df)}")

# 2. Khởi tạo và fit các Encoder cho biến chữ
job_encoder = LabelEncoder()
location_encoder = LabelEncoder()

df['job_title_encoded'] = job_encoder.fit_transform(df['job_title'])
df['company_location_encoded'] = location_encoder.fit_transform(df['company_location'])

# 3. Chọn tập đặc trưng và nhãn
X = df[["work_year", "job_title_encoded", "company_location_encoded"]]
y = df["salary_in_usd"]

print("Đang huấn luyện mô hình Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Lưu mô hình và các encoder vào thư mục models/
joblib.dump(model, os.path.join(models_dir, "salary_model.pkl"))
joblib.dump(job_encoder, os.path.join(models_dir, "job_encoder.pkl"))
joblib.dump(location_encoder, os.path.join(models_dir, "location_encoder.pkl"))

print("Đã cập nhật và lưu thành công các file .pkl với đầy đủ quốc gia và 5 năm gần đây!")