import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    print("Đang huấn luyện mô hình với đầy đủ các tính năng...")

    # Đã bổ sung 'Fresher' vào cấp độ kinh nghiệm
    df = pd.DataFrame({
        'work_year': [2022, 2023, 2024, 2022, 2023, 2024, 2023, 2024, 2022, 2023],
        'experience_level': ['Fresher', 'Junior', 'Mid-level', 'Senior', 'Fresher', 'Junior', 'Senior', 'Mid-level', 'Junior', 'Fresher'],
        'job_title': [
            'Front-end Developer', 'Back-end Developer', 'Mobile Developer', 
            'AI Engineer', 'Front-end Developer', 'Back-end Developer', 
            'Mobile Developer', 'AI Engineer', 'Front-end Developer', 'AI Engineer'
        ],
        'company_location': [
            'Vietnam', 'USA', 'UK', 'France', 'Singapore', 
            'Vietnam', 'UK', 'France', 'USA', 'Singapore'
        ],
        'salary': [30000, 45000, 90000, 135000, 32000, 48000, 140000, 95000, 50000, 35000]
    })

    encoders = {}
    categorical_cols = ['experience_level', 'job_title', 'company_location']

    # Huấn luyện và lưu encoder cho từng cột phân loại
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
        
        # Lưu lại file encoder
        encoder_path = os.path.join(model_dir, f"{col}_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"-> Đã lưu: {encoder_path}")

    # Chuẩn bị dữ liệu train mô hình chính
    X = df[['work_year']].copy()
    for col in categorical_cols:
        X[f'{col}_encoded'] = encoders[col].transform(df[col].astype(str))
    
    y = df['salary']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # Lưu mô hình chính
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"-> Đã lưu mô hình chính tại: {model_path}")

    print("Hoàn tất huấn luyện toàn bộ tính năng!")

if __name__ == '__main__':
    main()