import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# 데이터 로드
df = pd.read_csv("../raw/transaction_joined_balanced.csv")

# 금액형 컬럼 전처리
def clean_money(value):
    if pd.isna(value): return value
    value = str(value).strip().replace('$', '').replace(',', '').replace(' ', '')
    if value.startswith('(') and value.endswith(')'):
        value = value[1:-1]
    try:
        return float(value)
    except:
        return np.nan

money_cols = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
for col in money_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_money)

# 날짜형 컬럼 변환
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['acct_open_date'] = pd.to_datetime(df['acct_open_date'], errors='coerce')

# 카드 만료일을 해당 월의 말일로 변환
def convert_expires_to_last_day(expires_str):
    try:
        date = pd.to_datetime(expires_str, format='%m/%Y', errors='coerce')
        if pd.isna(date):
            return pd.NaT
        next_month = date + pd.offsets.MonthBegin(1)
        last_day = next_month - pd.Timedelta(days=1)
        return last_day
    except:
        return pd.NaT

df['expires_last_day'] = df['expires'].apply(convert_expires_to_last_day)

# 파생 변수 생성
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['account_age_days'] = (df['date'] - df['acct_open_date']).dt.days
df['months_to_expiry'] = (df['expires_last_day'].dt.year - df['date'].dt.year) * 12 + (df['expires_last_day'].dt.month - df['date'].dt.month)
df['transaction_age'] = df['date'].dt.year - df['birth_year']
df['years_to_retirement'] = df['retirement_age'] - df['transaction_age']
df['zip_prefix'] = df['zip'].astype(str).str[:3]
df['merchant_region'] = df['merchant_city'].astype(str) + '_' + df['merchant_state'].astype(str)

# 결측값 처리
df.fillna("Missing", inplace=True)

# 불필요/중복/식별자 컬럼 삭제
drop_cols = [
    'id', 'client_id', 'card_id', 'merchant_id', 'client_id_card',
    'expires', 'acct_open_date', 'birth_year', 'birth_month',
    'current_age', 'retirement_age', 'card_on_dark_web', 'mcc',
    'zip', 'merchant_city', 'merchant_state', 'expires_last_day', 'date'
]
drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)

# 범주형 변수 인코딩
cat_cols = [
    'use_chip', 'errors', 'mcc_type', 'gender', 'address', 'card_brand',
    'card_type', 'has_chip', 'zip_prefix', 'merchant_region'
]
for col in cat_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# feature, label 분리
X = df.drop(columns=['fraud'])
y = df['fraud']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# XGBoost 모델 훈련
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='logloss'
)
model.fit(X_train, y_train)

# SHAP 분석
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# SHAP 시각화
shap.summary_plot(
    shap_values, features=X_test, feature_names=X.columns,
    plot_type='bar', max_display=X.shape[1]
)

# SHAP 중요도 수치 출력 및 저장
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values(by='mean_abs_shap', ascending=False)

print("\n[SHAP Feature Importance - 상위 영향도 속성 순서]")
print(shap_importance)

shap_importance.to_csv("../raw/shap_feature_importance.csv", index=False)
