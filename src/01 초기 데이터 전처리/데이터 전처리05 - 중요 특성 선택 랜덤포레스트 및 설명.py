import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np


# 1. 데이터 로딩
data = pd.read_csv('../raw/transaction_joined_balance_feature_preprocess.csv')


# 4. 범주형 변수 인코딩
categorical_cols = [
    'client_id', 'merchant_id',
    'use_chip', 'errors', 'mcc', 'gender', 'address', 'card_brand',
    'card_type', 'has_chip', 'zip_prefix', 'merchant_state', 'merchant_city', 'expires_last_day'
]
for col in categorical_cols:
    if col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# 5. 수치형 변수 정규화는 트리 기반 모델은 불필요
'''num_cols = [
    'amount', 'latitude', 'longitude',
    'per_capita_income', 'yearly_income', 'total_debt', 'credit_score',
    'num_credit_cards', 'num_cards_issued', 'credit_limit',
    'year_pin_last_changed', 'hour', 'dayofweek', 'account_age_days'
]
for col in num_cols:
    if col not in data.columns:
        num_cols.remove(col)
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])'''


X = data.drop(columns=['fraud'])
y = data['fraud']

# 7. 학습/평가 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. 랜덤포레스트 모델 학습 및 중요도 산출
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
feature_names = X.columns

# 9. 중요도 높은 feature 정렬 및 출력
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("=== Fraud 탐지에 중요한 속성 (중요도 순) ===")
for feat, imp in feat_imp:
    print(f"{feat}: {imp:.4f}")

# 데이터프레임 변환
feat_imp_df = pd.DataFrame(feat_imp, columns=['feature', 'importance'])

# CSV로 저장
feat_imp_df.to_csv('../raw/random_forest_feature_importance.csv', index=False)


'''
# 10. 시각화
# 1. credit_score와 fraud 비율
plt.figure(figsize=(8, 6))
sns.boxplot(x='fraud', y='credit_score', data=data)
plt.title('Credit Score vs Fraud')
plt.xlabel('Fraud (0: No, 1: Yes)')
plt.ylabel('Credit Score')
plt.show()

# 2. amount와 fraud 비율
plt.figure(figsize=(8, 6))
sns.boxplot(x='fraud', y='amount', data=data)
plt.title('Amount vs Fraud')
plt.xlabel('Fraud (0: No, 1: Yes)')
plt.ylabel('Transaction Amount')
plt.show()

# 3. account_age_days와 fraud 비율
plt.figure(figsize=(8, 6))
sns.boxplot(x='fraud', y='account_age_days', data=data)
plt.title('Account Age (in days) vs Fraud')
plt.xlabel('Fraud (0: No, 1: Yes)')
plt.ylabel('Account Age (days)')
plt.show()

# 4. errors와 fraud 비율 (에러 유형별로 분포)
plt.figure(figsize=(10, 6))
sns.countplot(x='errors', hue='fraud', data=data)
plt.title('Errors vs Fraud')
plt.xlabel('Error Type')
plt.ylabel('Count')
plt.legend(title='Fraud', loc='upper right', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()
'''