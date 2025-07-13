import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. 데이터 로딩
data = pd.read_csv('../raw/transaction_labeled_balanced.csv')

# 2. 날짜 타입 변환 및 파생 변수 생성
data['date'] = pd.to_datetime(data['date'])
data['acct_open_date'] = pd.to_datetime(data['acct_open_date'])
data['hour'] = data['date'].dt.hour
data['dayofweek'] = data['date'].dt.dayofweek
data['account_age_days'] = (data['date'] - data['acct_open_date']).dt.days

# acct_open_date 컬럼 삭제 (파생 변수로 대체)
data = data.drop(columns=['acct_open_date'])

# 3. 범주형 변수 인코딩
categorical_cols = [
    'use_chip', 'merchant_city', 'merchant_state', 'errors', 'mcc_type',
    'gender', 'card_brand', 'card_type', 'has_chip', 'card_on_dark_web'
]
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# 4. 수치형 변수 정규화
num_cols = [
    'amount', 'current_age', 'retirement_age', 'latitude', 'longitude',
    'per_capita_income', 'yearly_income', 'total_debt', 'credit_score',
    'num_credit_cards', 'num_cards_issued', 'credit_limit',
    'year_pin_last_changed', 'hour', 'dayofweek', 'account_age_days'
]
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# 5. feature, label 분리 (식별자/날짜/연결용 컬럼 제외)
X = data.drop(columns=[
    'id', 'date', 'client_id', 'card_id', 'merchant_id', 'fraud'
])
y = data['fraud']

# 6. 학습/평가 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. 랜덤포레스트 모델 학습 및 중요도 산출
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
feature_names = X.columns

# 8. 중요도 높은 feature 정렬 및 출력
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("=== Fraud 탐지에 중요한 속성 (중요도 순) ===")
for feat, imp in feat_imp:
    print(f"{feat}: {imp:.4f}")
