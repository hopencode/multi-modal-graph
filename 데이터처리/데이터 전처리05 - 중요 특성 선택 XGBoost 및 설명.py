import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# 데이터 로드
df = pd.read_csv("../raw/transaction_joined_balance_feature_preprocess.csv")


# 범주형 변수 인코딩
cat_cols = [
    'client_id', 'merchant_id',
    'use_chip', 'errors', 'mcc_type', 'gender', 'address', 'card_brand',
    'card_type', 'has_chip', 'zip_prefix', 'merchant_state', 'merchant_city', 'expires_last_day'
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

# SHAP 중요도 수치 출력
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values(by='mean_abs_shap', ascending=False)

print("\n[SHAP Feature Importance - 상위 영향도 속성 순서]")
print(shap_importance)

shap_importance.to_csv("../raw/shap_feature_importance.csv", index=False)
