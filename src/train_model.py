import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "cleaned_salary_data.csv")
models_dir = os.path.join(current_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Mã hóa nhãn (Học toàn bộ dữ liệu quốc gia và chức vụ)
job_encoder = LabelEncoder()
location_encoder = LabelEncoder()

df["job_title_encoded"] = job_encoder.fit_transform(df["job_title"])
df["company_location_encoded"] = location_encoder.fit_transform(df["company_location"])

X = df[["work_year", "job_title_encoded", "company_location_encoded"]]
y = df["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- ĐANG HUẤN LUYỆN MÔ HÌNH ---")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print(f"Random Forest MAE: {mean_absolute_error(y_test, rf_pred)}")

# Lưu mô hình và các encoder
joblib.dump(rf, os.path.join(models_dir, "salary_model.pkl"))
joblib.dump(job_encoder, os.path.join(models_dir, "job_encoder.pkl"))
joblib.dump(location_encoder, os.path.join(models_dir, "location_encoder.pkl"))

print(f"\nĐã huấn luyện và lưu thành công các file mô hình vào thư mục: {models_dir}")