import os
import joblib
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
models_dir = os.path.join(root_dir, "models")

try:
    model = joblib.load(os.path.join(models_dir, "salary_model.pkl"))
    job_encoder = joblib.load(os.path.join(models_dir, "job_encoder.pkl"))
    location_encoder = joblib.load(os.path.join(models_dir, "location_encoder.pkl"))
    print("Đã tải mô hình và các encoder thành công!\n")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

print("--- HỆ THỐNG DỰ ĐOÁN MỨC LƯƠNG LẬP TRÌNH VIÊN TOÀN CẦU ---")

print("\n[Các nhóm vị trí công việc có sẵn]:")
print(list(job_encoder.classes_))

print("\n[Gợi ý một số mã quốc gia (Location) mẫu]:")
print(list(location_encoder.classes_[:10]))
print("-" * 60)

try:
    work_year = int(input("Nhập năm làm việc (ví dụ 2026): "))
    job_title = input("Nhập tên vị trí (chọn đúng tên trong danh sách trên): ").strip()
    company_location = input("Nhập mã quốc gia (ví dụ US, VN, GB...): ").strip().upper()

    # Chuyển đổi chữ thành số bằng Encoder
    encoded_job = job_encoder.transform([job_title])[0]
    encoded_location = location_encoder.transform([company_location])[0]

    # Tạo DataFrame đúng 3 cột khớp với lúc train model
    X_input = pd.DataFrame(
        [[work_year, encoded_job, encoded_location]],
        columns=["work_year", "job_title_encoded", "company_location_encoded"]
    )

    # Dự đoán mức lương (USD)
    predicted_salary_usd = model.predict(X_input)[0]

    # In kết quả trực quan chuẩn quốc tế (USD)
    print("\n" + "="*50)
    print(f"       KẾT QUẢ DỰ ĐOÁN MỨC LƯƠNG TOÀN CẦU       ")
    print("="*50)
    print(f" 🔹 Năm làm việc: {work_year}")
    print(f" 🔹 Vị trí công việc: {job_title}")
    print(f" 🔹 Khu vực (Quốc gia): {company_location}")
    print("-" * 50)
    print(f" ➔ Mức lương dự kiến: ${predicted_salary_usd:,.2f} USD / năm")
    print("="*50)

except Exception as ex:
    print(f"\n[Lỗi]: {ex}. Bạn hãy đảm bảo nhập đúng chính tả tên vị trí và mã quốc gia có trong danh sách gợi ý nhé!")