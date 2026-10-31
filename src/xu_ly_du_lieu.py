import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Global IT Salary Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")

RAW_PATH = os.path.join(DATA_DIR, "salary_data.txt")
CLEAN_PATH = os.path.join(DATA_DIR, "cleaned_salary_data.csv")


# =====================================================
# 2. LANGUAGE SYSTEM
# =====================================================

TEXT = {
    "VI": {
        "app_name": "Global IT Salary",
        "subtitle": "Hệ thống dự đoán mức lương ngành Công nghệ thông tin toàn cầu",
        "home": "Trang chủ",
        "prediction": "Dự đoán mức lương",
        "analysis": "Phân tích dữ liệu",
        "dataset": "Dữ liệu",
        "settings": "Cài đặt",
        "overview": "Tổng quan dữ liệu",
        "total": "Tổng số dữ liệu",
        "countries": "Quốc gia",
        "jobs": "Nhóm nghề nghiệp",
        "predict_title": "Dự đoán mức lương IT",
        "predict_desc": "Nhập thông tin công việc để dự đoán mức lương.",
        "country": "Quốc gia làm việc",
        "job": "Vị trí công việc",
        "experience": "Số năm kinh nghiệm",
        "level": "Cấp bậc",
        "employment": "Hình thức làm việc",
        "remote": "Tỷ lệ làm việc từ xa (%)",
        "company_size": "Quy mô công ty",
        "predict": "Dự đoán mức lương",
        "result": "Kết quả dự đoán",
        "salary": "Mức lương dự đoán",
        "monthly": "Mức lương trung bình tháng",
        "yearly": "Mức lương dự kiến năm",
        "analysis_title": "Phân tích thị trường IT",
        "salary_country": "Mức lương theo quốc gia",
        "salary_job": "Mức lương theo vị trí",
        "salary_exp": "Mức lương theo kinh nghiệm",
        "dataset_title": "Dữ liệu sau khi xử lý",
        "search": "Tìm kiếm dữ liệu",
        "no_data": "Không tìm thấy dữ liệu phù hợp.",
        "model": "Đánh giá mô hình",
        "mae": "Sai số tuyệt đối trung bình (MAE)",
        "rmse": "Căn bậc hai sai số bình phương trung bình (RMSE)",
        "r2": "R² Score",
        "home_desc": "Khám phá và dự đoán mức lương ngành IT tại nhiều quốc gia trên thế giới.",
        "warning": "Mô hình chưa thể dự đoán do dữ liệu không đủ.",
        "salary_distribution": "Phân phối mức lương",
        "currency": "Đơn vị tiền tệ địa phương",
        "model_info": "Mô hình sử dụng Random Forest Regression."
    },

    "EN": {
        "app_name": "Global IT Salary",
        "subtitle": "Global Information Technology Salary Prediction System",
        "home": "Home",
        "prediction": "Salary Prediction",
        "analysis": "Data Analysis",
        "dataset": "Dataset",
        "settings": "Settings",
        "overview": "Dataset Overview",
        "total": "Total Records",
        "countries": "Countries",
        "jobs": "Job Categories",
        "predict_title": "IT Salary Prediction",
        "predict_desc": "Enter your job information to estimate your salary.",
        "country": "Work Country",
        "job": "Job Position",
        "experience": "Years of Experience",
        "level": "Seniority Level",
        "employment": "Employment Type",
        "remote": "Remote Work Ratio (%)",
        "company_size": "Company Size",
        "predict": "Predict Salary",
        "result": "Prediction Result",
        "salary": "Predicted Salary",
        "monthly": "Estimated Monthly Salary",
        "yearly": "Estimated Annual Salary",
        "analysis_title": "IT Market Analysis",
        "salary_country": "Salary by Country",
        "salary_job": "Salary by Job Position",
        "salary_exp": "Salary by Experience",
        "dataset_title": "Processed Dataset",
        "search": "Search Dataset",
        "no_data": "No matching data found.",
        "model": "Model Evaluation",
        "mae": "Mean Absolute Error (MAE)",
        "rmse": "Root Mean Squared Error (RMSE)",
        "r2": "R² Score",
        "home_desc": "Explore and predict IT salaries across countries worldwide.",
        "warning": "The model cannot predict due to insufficient data.",
        "salary_distribution": "Salary Distribution",
        "currency": "Local Currency",
        "model_info": "Model: Random Forest Regression."
    }
}


# =====================================================
# 3. COUNTRY AND CURRENCY
# =====================================================

COUNTRIES = {
    "Vietnam": {"code": "VN", "currency": "VND", "rate": 26000, "symbol": "₫"},
    "United States": {"code": "US", "currency": "USD", "rate": 1, "symbol": "$"},
    "United Kingdom": {"code": "GB", "currency": "GBP", "rate": 0.75, "symbol": "£"},
    "Japan": {"code": "JP", "currency": "JPY", "rate": 150, "symbol": "¥"},
    "Singapore": {"code": "SG", "currency": "SGD", "rate": 1.30, "symbol": "S$"},
    "Germany": {"code": "DE", "currency": "EUR", "rate": 0.92, "symbol": "€"},
    "Canada": {"code": "CA", "currency": "CAD", "rate": 1.38, "symbol": "C$"},
    "Australia": {"code": "AU", "currency": "AUD", "rate": 1.52, "symbol": "A$"},
    "South Korea": {"code": "KR", "currency": "KRW", "rate": 1400, "symbol": "₩"},
    "India": {"code": "IN", "currency": "INR", "rate": 86, "symbol": "₹"}
}

COUNTRY_ALIASES = {
    "VN": "VN", "VIETNAM": "VN", "VIET NAM": "VN",
    "US": "US", "USA": "US", "UNITED STATES": "US",
    "GB": "GB", "UK": "GB", "UNITED KINGDOM": "GB",
    "JP": "JP", "JAPAN": "JP",
    "SG": "SG", "SINGAPORE": "SG",
    "DE": "DE", "GERMANY": "DE",
    "CA": "CA", "CANADA": "CA",
    "AU": "AU", "AUSTRALIA": "AU",
    "KR": "KR", "SOUTH KOREA": "KR",
    "IN": "IN", "INDIA": "IN"
}


# =====================================================
# 4. JOB CLASSIFICATION
# =====================================================

def map_job_title(title):

    title = str(title).lower()

    if any(k in title for k in [
        "android", "ios", "flutter", "mobile",
        "react native", "swift", "kotlin"
    ]):
        return "Mobile Engineer"

    elif any(k in title for k in [
        "machine learning", "deep learning",
        "artificial intelligence", "data scientist",
        "data science", "nlp", "computer vision",
        "ml engineer", "ai engineer"
    ]) or re.search(r"\b(ai|ml)\b", title):
        return "AI Engineer"

    elif any(k in title for k in [
        "front-end", "frontend", "front end",
        "react developer", "vue", "angular",
        "ui developer", "web designer"
    ]):
        return "Front-end Engineer"

    elif any(k in title for k in [
        "back-end", "backend", "back end",
        "server", "api developer", "java developer",
        "python developer", "node.js"
    ]):
        return "Back-end Engineer"

    return "Other"


# =====================================================
# 5. LOAD AND CLEAN DATA
# =====================================================

@st.cache_data
def load_data():

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Cannot find dataset: {RAW_PATH}"
        )

    df = pd.read_csv(
        RAW_PATH,
        sep=None,
        engine="python"
    )

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Standardize common column names
    aliases = {
        "job_title": ["job_title", "title", "position"],
        "country": [
            "company_location", "country",
            "location", "work_country"
        ],
        "experience_level": [
            "experience_level", "seniority",
            "experience", "level"
        ],
        "salary_in_usd": [
            "salary_in_usd", "salary_usd",
            "annual_salary_usd"
        ],
        "salary": ["salary", "annual_salary"],
        "salary_currency": [
            "salary_currency", "currency"
        ],
        "employment_type": [
            "employment_type", "employment"
        ],
        "remote_ratio": ["remote_ratio"],
        "company_size": ["company_size"]
    }

    for standard, alternatives in aliases.items():
        if standard not in df.columns:
            for col in alternatives:
                if col in df.columns:
                    df.rename(
                        columns={col: standard},
                        inplace=True
                    )
                    break

    required = ["job_title", "country"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Dataset is missing required column: {col}"
            )

    # Prefer salary standardized in USD
    if "salary_in_usd" not in df.columns:

        if "salary" not in df.columns:
            raise ValueError(
                "Dataset must contain salary_in_usd "
                "or salary and salary_currency."
            )

        if "salary_currency" not in df.columns:
            raise ValueError(
                "Cannot convert salary to USD because "
                "salary_currency is missing."
            )

        currencies = df["salary_currency"].astype(str).str.upper()

        if not currencies.eq("USD").all():
            raise ValueError(
                "Mixed currencies detected. Please provide "
                "salary_in_usd for reliable international prediction."
            )

        df["salary_in_usd"] = df["salary"]

    # Convert salary to numeric
    df["salary_in_usd"] = pd.to_numeric(
        df["salary_in_usd"],
        errors="coerce"
    )

    df.dropna(
        subset=["job_title", "country", "salary_in_usd"],
        inplace=True
    )

    # Remove invalid salary
    df = df[
        (df["salary_in_usd"] > 0) &
        (df["salary_in_usd"] < 1000000)
    ].copy()

    # Normalize job categories
    df["job_category"] = df["job_title"].apply(
        map_job_title
    )

    # Normalize country codes
    df["country"] = (
        df["country"].astype(str).str.strip().str.upper()
    )

    df["country"] = df["country"].apply(
        lambda x: COUNTRY_ALIASES.get(x, x)
    )

    # Experience level
    if "experience_level" not in df.columns:
        df["experience_level"] = "Unknown"

    level_map = {
        "EN": "Entry-level",
        "MI": "Mid-level",
        "SE": "Senior",
        "EX": "Executive"
    }

    df["experience_level"] = (
        df["experience_level"]
        .astype(str).str.upper().replace(level_map)
    )

    # Optional fields
    if "employment_type" not in df.columns:
        df["employment_type"] = "Unknown"

    if "remote_ratio" not in df.columns:
        df["remote_ratio"] = 0

    if "company_size" not in df.columns:
        df["company_size"] = "Unknown"

    df["remote_ratio"] = pd.to_numeric(
        df["remote_ratio"],
        errors="coerce"
    ).fillna(0)

    df.drop_duplicates(inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(CLEAN_PATH, index=False)

    return df


# =====================================================
# 6. TRAIN MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def train_model():

    df = load_data()

    features = [
        "job_category",
        "country",
        "experience_level",
        "employment_type",
        "remote_ratio",
        "company_size"
    ]

    target = "salary_in_usd"

    X = df[features].copy()
    y = df[target].copy()

    categorical_features = [
        "job_category",
        "country",
        "experience_level",
        "employment_type",
        "company_size"
    ]

    numerical_features = ["remote_ratio"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocess),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    if len(df) < 20:
        raise ValueError(
            "At least 20 valid records are required."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, df, metrics


# =====================================================
# 7. INITIALIZE APPLICATION
# =====================================================

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
    }

    .sub-title {
        color: #808080;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


if "language" not in st.session_state:
    st.session_state.language = "VI"

with st.sidebar:

    st.markdown("## 🌍 Global IT Salary")

    st.selectbox(
        "🌐 Language / Ngôn ngữ",
        ["VI", "EN"],
        key="language"
    )

    lang = st.session_state.language
    t = TEXT[lang]

    st.divider()

    page = st.radio(
        "Navigation",
        [
            t["home"],
            t["prediction"],
            t["analysis"],
            t["dataset"],
            t["settings"]
        ]
    )

    st.divider()

    st.caption("Global IT Salary Prediction")
    st.caption("Machine Learning Project")


# =====================================================
# 8. LOAD MODEL AND DATA
# =====================================================

try:
    with st.spinner("Loading dataset and training model..."):
        model, df, metrics = train_model()

except Exception as error:

    st.error(f"Application error: {error}")

    st.info(
        "Check your dataset columns and make sure "
        "salary_in_usd is available."
    )

    st.stop()


# =====================================================
# 9. HOME PAGE
# =====================================================

if page == t["home"]:

    st.markdown(
        f'<div class="main-title">🌍 {t["app_name"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sub-title">{t["subtitle"]}</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.title(t["overview"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            t["total"],
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            t["countries"],
            df["country"].nunique()
        )

    with col3:
        st.metric(
            t["jobs"],
            df["job_category"].nunique()
        )

    st.divider()

    st.subheader(t["home"])

    st.write(t["home_desc"])

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "🤖 Machine Learning\n\n"
            "Random Forest Regression"
        )

    with col2:
        st.info(
            "🌎 Global Market\n\n"
            "Multi-country salary analysis"
        )

    st.subheader(t["model"])

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"${metrics['MAE']:,.0f}")
    c2.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    c3.metric("R²", f"{metrics['R2']:.3f}")


# =====================================================
# 10. SALARY PREDICTION
# =====================================================

elif page == t["prediction"]:

    st.title(f"💰 {t['predict_title']}")

    st.write(t["predict_desc"])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        country_name = st.selectbox(
            t["country"],
            list(COUNTRIES.keys())
        )

        job = st.selectbox(
            t["job"],
            [
                "Front-end Engineer",
                "Back-end Engineer",
                "Mobile Engineer",
                "AI Engineer",
                "Other"
            ]
        )

        experience = st.number_input(
            t["experience"],
            min_value=0,
            max_value=40,
            value=2
        )

    with col2:

        level = st.selectbox(
            t["level"],
            [
                "Entry-level",
                "Mid-level",
                "Senior",
                "Executive"
            ]
        )

        employment = st.selectbox(
            t["employment"],
            ["FT", "PT", "CT", "FL"]
        )

        remote = st.slider(
            t["remote"],
            0, 100, 0
        )

    company_size = st.selectbox(
        t["company_size"],
        ["S", "M", "L"]
    )

    if st.button(
        f"🚀 {t['predict']}",
        type="primary",
        use_container_width=True
    ):

        country_code = COUNTRIES[country_name]["code"]

        # Check whether selected country exists in training data
        if country_code not in df["country"].values:

            st.warning(
                "This country is not represented in the training "
                "dataset. The model may not provide a reliable estimate."
            )

        input_data = pd.DataFrame([{
            "job_category": job,
            "country": country_code,
            "experience_level": level,
            "employment_type": employment,
            "remote_ratio": remote,
            "company_size": company_size
        }])

        prediction = model.predict(input_data)[0]

        currency = COUNTRIES[country_name]

        local_salary = prediction * currency["rate"]

        monthly_salary = prediction / 12

        st.divider()

        st.subheader(f"🎯 {t['result']}")

        st.metric(
            t["yearly"],
            f"${prediction:,.0f} USD"
        )

        st.metric(
            t["monthly"],
            f"${monthly_salary:,.0f} USD"
        )

        st.metric(
            f"{t['salary']} ({currency['currency']})",
            f"{currency['symbol']}{local_salary:,.0f}"
        )

        st.caption(t["model_info"])

        st.info(
            "The result is an estimate based on the training "
            "dataset, not a guaranteed market salary."
        )


# =====================================================
# 11. DATA ANALYSIS
# =====================================================

elif page == t["analysis"]:

    st.title(f"📊 {t['analysis_title']}")

    st.divider()

    # Country salary chart
    st.subheader(t["salary_country"])

    country_chart = (
        df.groupby("country")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    country_chart.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Country")
    ax.set_title("Average IT Salary by Country")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Job category chart
    st.subheader(t["salary_job"])

    job_chart = (
        df.groupby("job_category")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    job_chart.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_title("Average Salary by Job Category")

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Experience analysis
    st.subheader(t["salary_exp"])

    exp_chart = (
        df.groupby("experience_level")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    exp_chart.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_title("Average Salary by Experience Level")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Salary distribution
    st.subheader(t["salary_distribution"])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["salary_in_usd"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Annual Salary (USD)")
    ax.set_ylabel("Frequency")
    ax.set_title("IT Salary Distribution")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# 12. DATASET PAGE
# =====================================================

elif page == t["dataset"]:

    st.title(f"📁 {t['dataset_title']}")

    st.write(
        f"{len(df):,} records | "
        f"{len(df.columns)} columns"
    )

    search = st.text_input(t["search"])

    if search:

        mask = df.astype(str).apply(
            lambda col: col.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(axis=1)

        display_df = df[mask]

    else:
        display_df = df

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500
    )

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_salary_data.csv",
        mime="text/csv"
    )


# =====================================================
# 13. SETTINGS PAGE
# =====================================================

elif page == t["settings"]:

    st.title(f"⚙️ {t['settings']}")

    st.subheader("Machine Learning")

    st.write("Algorithm: Random Forest Regression")

    st.write("Training/Test split: 80% / 20%")

    st.write("Random State: 42")

    st.divider()

    st.subheader(t["model"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        t["mae"],
        f"${metrics['MAE']:,.2f}"
    )

    col2.metric(
        t["rmse"],
        f"${metrics['RMSE']:,.2f}"
    )

    col3.metric(
        t["r2"],
        f"{metrics['R2']:.4f}"
    )

    st.divider()

    st.write(
        "Data source: Local salary dataset."
    )

    st.write(
        "Model: Random Forest Regressor."
    )

    st.write(
        "Target variable: Annual Salary (USD)."
    )
