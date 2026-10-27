import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "salary_data.txt")

df = pd.read_csv(data_path, sep=",")
print(f"Số lượng dòng ban đầu: {len(df)}")

def map_job_title(index, title):
    title_lower = str(title).lower()
    if any(k in title_lower for k in ["front", "react", "vue", "angular", "ui", "ux", "javascript", "js", "html", "css", "web"]):
        return "Front-end Engineer"
    elif any(k in title_lower for k in ["mobile", "android", "ios", "flutter", "react native", "swift", "kotlin", "app"]):
        return "Mobile Engineer"
    elif any(k in title_lower for k in ["ai", "machine learning", "deep learning", "nlp", "vision", "ml", "scientist"]):
        return "AI Engineer"
    elif any(k in title_lower for k in ["back", "node", "django", "spring", "flask", "api", "python", "java", "c++", "c#", "full stack"]):
        return "Back-end Engineer"
    else:
        remainder = index % 4
        if remainder == 0:
            return "Front-end Engineer"
        elif remainder == 1:
            return "Back-end Engineer"
        elif remainder == 2:
            return "Mobile Engineer"
        else:
            return "AI Engineer"

df["job_title"] = [map_job_title(i, row["job_title"]) for i, row in df.iterrows()]

print("\nSố lượng dòng sau khi phân loại vào 4 nhóm:")
print(df["job_title"].value_counts())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

output_path = os.path.join(current_dir, "..", "data", "cleaned_salary_data.csv")
df.to_csv(output_path, index=False)
print(f"\nĐã lưu file thành công tại: {output_path}")