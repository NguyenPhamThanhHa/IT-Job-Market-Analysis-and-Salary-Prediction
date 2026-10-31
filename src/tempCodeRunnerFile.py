import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi về thư mục gốc "Python cuối kì" nếu file nằm trong thư mục src
root_dir = os.path.dirname(current_dir)
data_path = os.path.join(root_dir, "data", "cleaned_salary_data.csv")

# Đọc dữ liệu đã làm sạch
df = pd.read_csv(data_path)

# Cấu hình giao diện biểu đồ đẹp mắt
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# --- Biểu đồ 1: Phân phối mức lương theo USD (salary_in_usd) ---
plt.subplot(1, 2, 1)
sns.histplot(df["salary_in_usd"], bins=30, kde=True, color="skyblue")
plt.title("Phân phối mức lương (USD)")
plt.xlabel("Lương (USD)")
plt.ylabel("Số lượng")

# --- Biểu đồ 2: Top 5 Job Title có mức lương trung bình cao nhất ---
plt.subplot(1, 2, 2)
top_jobs = df.groupby("job_title")["salary_in_usd"].mean().nlargest(5).reset_index()
sns.barplot(data=top_jobs, x="salary_in_usd", y="job_title", palette="viridis")
plt.title("Top 5 công việc có lương cao nhất")
plt.xlabel("Lương trung bình (USD)")
plt.ylabel("Tên công việc")

plt.tight_layout()

# Hiển thị biểu đồ
print("Đang hiển thị biểu đồ trực quan hóa dữ liệu...")
plt.show()