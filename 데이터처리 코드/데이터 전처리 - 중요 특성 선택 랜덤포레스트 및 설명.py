import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

# 카드 만료일 전처리
def convert_expires_to_last_day(expires_str):
    try:
        # MM/YYYY → datetime 변환 (해당 월 1일)
        date = pd.to_datetime(expires_str, format='%m/%Y', errors='coerce')
        if pd.isna(date):
            return pd.NaT
        # 다음 달 1일에서 하루 빼서 말일 계산
        next_month = date + pd.offsets.MonthBegin(1)
        last_day = next_month - pd.Timedelta(days=1)
        return last_day
    except:
        return pd.NaT


def months_diff(d1, d2):
    if pd.isna(d1) or pd.isna(d2):
        return pd.NA
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)

# 1. 데이터 로딩
data = pd.read_csv('../raw/transaction_joined_balanced.csv')

# 2. 금액형 컬럼(문자열 포함 가능성) float 변환
money_cols = [
    'amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit'
]
for col in money_cols:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .replace(r'[\$,()]', '', regex=True)
            .replace('', '0')
            .astype(float)
        )

# 3. 날짜 타입 변환 및 파생 변수 생성
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['acct_open_date'] = pd.to_datetime(data['acct_open_date'], errors='coerce')

# 파생 변수 생성
data['hour'] = data['date'].dt.hour
data['dayofweek'] = data['date'].dt.dayofweek
data['account_age_days'] = (data['date'] - data['acct_open_date']).dt.days
#data['expires'] = data['expires'].apply(convert_expires_to_last_day)
data['expires_last_day'] = data['expires'].apply(convert_expires_to_last_day)
data['months_to_expiry'] = (data['expires_last_day'].dt.year - data['date'].dt.year) * 12 + (data['expires_last_day'].dt.month - data['date'].dt.month)

# 거래 발생 시점의 고객 나이, 은퇴까지 남은 연도 계산
data['transaction_age'] = data['date'].dt.year - data['birth_year']
data['years_to_retirement'] = data['retirement_age'] - data['transaction_age']

# zip(우편번호) 앞자리 기준 클러스터링
# zip 앞 3자리 = Sectional Center Facility(SCF) 단위
# zip 앞 3자리는 시/군/구보다 넓고, 주(State)보다는 좁은 단위
data['zip_prefix'] = data['zip'].astype(str).str[:3]

# zip 대신 주와 도시 정보 조합
data['merchant_region'] = data['merchant_city'] + '_' + data['merchant_state']

# acct_open_date 컬럼 삭제 (파생 변수로 대체)
data = data.drop(columns=['acct_open_date'])

# 결측값 처리
data.fillna("Missing", inplace=True)

# 4. 범주형 변수 인코딩
categorical_cols = [
    'use_chip', 'errors', 'mcc_type', 'gender', 'address', 'card_brand',
    'card_type', 'has_chip', 'zip_prefix', 'merchant_region'
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

# 6. feature, label 분리 (식별자/날짜/연결용 컬럼 제외)
drop_cols = [
    'id', 'client_id', 'card_id', 'merchant_id', 'client_id_card',
    'expires', 'acct_open_date', 'birth_year', 'birth_month',
    'current_age', 'retirement_age', 'card_on_dark_web', 'mcc',
    'zip', 'merchant_city', 'merchant_state', 'expires_last_day',
    'date'
]
drop_cols = [col for col in drop_cols if col in data.columns]
data = data.drop(columns=drop_cols)
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