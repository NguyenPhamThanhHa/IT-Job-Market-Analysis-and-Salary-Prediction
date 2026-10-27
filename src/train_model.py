import os
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
    # Dữ liệu mẫu giả lập
    df = pd.DataFrame({
        "work_year": [2024, 2025, 2024, 2025, 2024],
        "job_title": ["Front End Developer", "Back End Engineer", "Mobile Developer", "AI Engineer", "Data Analyst"],
        "company_location": ["VN", "US", "US", "VN", "GB"],
        "salary_in_usd": [25000, 110000, 30000, 130000, 40000]
    })

print("Đang lọc dữ liệu cho các vị trí: Front End, Back End, Mobile, AI Engineer...")

# =====================================================================
# BỘ LỌC: Chỉ giữ lại các dòng có chứa từ khóa thuộc 4 nhóm công việc trên
# =====================================================================
# Dùng biểu thức chứa từ khóa (không phân biệt hoa thường)
import re
pattern = r'front|back|mobile|ai|machine learning'
df = df[df['job_title'].str.contains(pattern, case=False, na=False)]

# Kiểm tra xem sau khi lọc còn dữ liệu không
if df.empty:
    print("Cảnh báo: Dữ liệu sau khi lọc bị trống! Hãy kiểm tra lại tên cột hoặc từ khóa.")
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

print("Đang huấn luyện mô hình Random Forest với dữ liệu đã lọc...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Lưu mô hình và các encoder mới vào thư mục models/
joblib.dump(model, os.path.join(models_dir, "salary_model.pkl"))
joblib.dump(job_encoder, os.path.join(models_dir, "job_encoder.pkl"))
joblib.dump(location_encoder, os.path.join(models_dir, "location_encoder.pkl"))

print("Đã cập nhật và lưu thành công các file .pkl mới!")