import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# 1. 데이터 로드
df = pd.read_csv("../raw/transaction_labeled_balanced.csv")

# 2. 금액 관련 전처리 함수
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

# 3. 날짜 처리: 거래일 및 계좌 개설일 전처리
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['acct_open_date'] = pd.to_datetime(df['acct_open_date'], errors='coerce')

# 거래 발생 시각 파생 변수
df['transaction_hour'] = df['date'].dt.hour
df['transaction_dayofweek'] = df['date'].dt.weekday

# 계좌 개설 후 경과일 (신규 고객 여부 판단 가능)
df['days_since_acct_open'] = (df['date'] - df['acct_open_date']).dt.days

# 날짜 관련 컬럼 제거
df.drop(columns=['date', 'acct_open_date'], inplace=True)

# 4. 결측값 처리 + 범주형 인코딩
df.fillna("Missing", inplace=True)
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 5. 입력(X), 출력(y) 설정
X = df.drop(columns=['fraud', 'id'])
y = df['fraud']

# 6. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 7. XGBoost 모델 훈련
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 8. SHAP 분석
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# 9. SHAP 시각화: 전체 속성 시각화
shap.summary_plot(shap_values, features=X_test, feature_names=X.columns, plot_type='bar', max_display=X.shape[1])

# 10. SHAP 평균 중요도 수치 계산 및 출력
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values(by='mean_abs_shap', ascending=False)

print("\n[SHAP Feature Importance - 상위 영향도 속성 순서]")
print(shap_importance)

# --- 11. 필요 시 CSV 저장 ---
shap_importance.to_csv("../raw/shap_feature_importance.csv", index=False)
