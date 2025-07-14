import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autofeat import AutoFeatClassifier
import warnings
import gc

warnings.filterwarnings("ignore")

# --- 데이터 로드 및 전처리 ---
# 1. read_csv 시 필요한 컬럼만 선택적으로 로드하여 초기 메모리 사용량 줄이기
#    (이 부분은 정확히 어떤 컬럼이 필요한지 알아야 최적화 가능.
#     지금은 모든 컬럼을 로드하되, 불필요한 컬럼을 나중에 바로 drop하는 방식)
df = pd.read_csv("../raw/transaction_joined_balanced.csv")
print(f"초기 데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
print(f"초기 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")


# clean_money 함수는 유지하되, 약간의 최적화 (str.strip() 위치 등)
def clean_money(value):
    if pd.isna(value): return value
    # str() 변환은 apply 내에서 문자열이 아닌 값에 대해 필요할 때만
    s_value = str(value) # value가 숫자형일 수 있으므로 str 변환
    s_value = s_value.strip().replace('$', '').replace(',', '')
    # 괄호 제거 로직은 실제 데이터에 괄호가 없다는 확인에 따라 제외 (이전 요청 반영)
    # if s_value.startswith('(') and s_value.endswith(')'):
    #     s_value = s_value[1:-1]
    try:
        return float(s_value)
    except ValueError: # 보다 구체적인 예외 처리
        return np.nan

money_cols = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
for col in money_cols:
    if col in df.columns:
        # apply 대신 벡터화된 연산을 시도 (가능하다면)
        # 하지만 clean_money처럼 복잡한 로직은 apply가 필요할 수 있음
        # 여기서는 apply 사용을 유지하되, 메모리 사용에 주의
        df[col] = df[col].apply(clean_money)
        # clean_money 후 NaN이 생긴 경우 평균으로 채우기 (아래 결측값 처리와 중복될 수 있음)
        # df[col].fillna(df[col].mean(), inplace=True) # 아래에서 일괄 처리되므로 여기서는 생략

# 날짜 컬럼을 datetime으로 변환
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['acct_open_date'] = pd.to_datetime(df['acct_open_date'], errors='coerce')

def convert_expires_to_last_day(expires_str):
    try:
        # format 지정으로 더 빠르게 변환 시도
        date = pd.to_datetime(expires_str, format='%m/%Y', errors='coerce')
        if pd.isna(date):
            return pd.NaT
        next_month = date + pd.offsets.MonthBegin(1)
        last_day = next_month - pd.Timedelta(days=1)
        return last_day
    except: # pd.to_datetime이 이미 errors='coerce'를 처리하므로 이 try-except는 불필요할 수 있음
        return pd.NaT

df['expires_last_day'] = df['expires'].apply(convert_expires_to_last_day)

# --- 파생 변수 생성 ---
# dt 접근자 사용으로 메모리 효율성 유지
df['hour'] = df['date'].dt.hour.astype('int8') # int8로 최적화
df['dayofweek'] = df['date'].dt.dayofweek.astype('int8') # int8로 최적화

# large timedelta64 to int
df['account_age_days'] = (df['date'] - df['acct_open_date']).dt.days.astype('int32') # int32로 최적화

# calculate_months_to_expiry 함수를 정의하여 NaN 처리 및 int 타입 변환을 명확히
def calculate_months_to_expiry(row_date, row_expires):
    if pd.isna(row_date) or pd.isna(row_expires):
        return np.nan
    return (row_expires.year - row_date.year) * 12 + (row_expires.month - row_date.month)

# apply 대신 벡터화된 연산을 사용하거나, apply를 사용할 경우 astype으로 최적화
df['months_to_expiry'] = df.apply(lambda row: calculate_months_to_expiry(row['date'], row['expires_last_day']), axis=1).astype('float32') # float32로 최적화

df['transaction_age'] = (df['date'].dt.year - df['birth_year']).astype('int16') # int16으로 최적화
df['years_to_retirement'] = (df['retirement_age'] - df['transaction_age']).astype('int16') # int16으로 최적화

# zip_prefix는 문자열이므로 category로 변환하기 전까지는 object.
# 나중에 drop되므로 여기서는 최적화 생략 가능
df['zip_prefix'] = df['zip'].astype(str).str[:3]

# merchant_region은 문자열 조합. 나중에 LabelEncoder에 의해 변환될 것임
df['merchant_region'] = df['merchant_city'].astype(str) + '_' + df['merchant_state'].astype(str)

print(f"파생 변수 생성 후 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")


# --- 불필요/중복/식별자 컬럼 삭제 ---
drop_cols = [
    'id', 'client_id', 'card_id', 'merchant_id', 'client_id_card',
    'expires', 'acct_open_date', 'birth_year', 'birth_month',
    'current_age', 'retirement_age', 'card_on_dark_web', 'mcc',
    'zip', 'merchant_city', 'merchant_state', 'expires_last_day', 'date'
]
drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)
print(f"불필요 컬럼 삭제 후 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
gc.collect() # 메모리 해제 강제


# --- 결측값 처리 ---
# 메모리 절약을 위해 inplace=True 사용
for col in df.columns:
    if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
        df[col].fillna("Missing", inplace=True)
    elif pd.api.types.is_numeric_dtype(df[col]):
        # 숫자형 결측값은 평균 대신 중앙값(median)으로 채울 수도 있다. (이상치 영향 감소)
        # 대규모 데이터에서는 mean 계산도 메모리를 사용하므로 주의
        df[col].fillna(df[col].mean(), inplace=True)
print(f"결측값 처리 후 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
gc.collect()


# --- 범주형 변수 인코딩 ---
cat_cols = [
    'use_chip', 'errors', 'mcc_type', 'gender', 'address', 'card_brand',
    'card_type', 'has_chip', 'zip_prefix', 'merchant_region'
]
for col in cat_cols:
    if col in df.columns:
        # LabelEncoder는 새 Series를 반환하므로, inplace=True는 안 됨
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df[col] = df[col].astype('int16') # LabelEncoder 결과는 int64이므로 int16/int32로 최적화
print(f"범주형 인코딩 후 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
gc.collect()


# --- Feature, Label 분리 ---
X = df.drop(columns=['fraud'])
y = df['fraud']

# 원본 DataFrame df는 더 이상 필요 없으므로 메모리에서 해제
del df
gc.collect()
print(f"X, y 분리 및 원본 df 해제 후 X 메모리 사용량: {X.memory_usage(deep=True).sum() / (1024**2):.2f} MB")


# --- 학습/테스트 데이터 분리 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 원본 X는 이제 필요 없으므로 해제 (메모리 절약)
del X, y
gc.collect()
print(f"학습/테스트 데이터 분리 후 X_train 메모리 사용량: {X_train.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print("원본 특성 개수:", X_train.shape[1])

### autofeat를 사용한 중요 특징 분석 및 파생 변수 생성

# `AutoFeatClassifier` 초기화 시 리소스 최적화
# 1. n_jobs: -1 (모든 코어)은 메모리 사용량을 크게 늘릴 수 있음.
#             메모리 부족 시에는 1 (단일 코어) 또는 CPU 코어 수의 절반 등으로 줄여볼 것.
# 2. feateng_steps: 이 값이 높을수록 생성되는 특징의 수가 기하급수적으로 늘어남.
#                   메모리 부족의 가장 큰 원인 중 하나이므로 1로 시작해보고 늘려갈 것.
# 3. max_n_features: 생성/선택할 최대 특징 수를 제한하여 메모리 사용량을 제어.
#                    (기본값은 1000000000000.0으로 사실상 무제한)
# 4. drop_most_highly_correlated: 매우 높은 상관 관계를 가진 특징 쌍 중 하나를 제거하여
#                                 특징 수를 줄이고 안정성 향상 (메모리 간접적 도움)
# 5. min_std: 표준편차가 작은 (변별력 낮은) 특징 제거
af = AutoFeatClassifier(verbose=2, feateng_steps=1, n_jobs=1) # 너무 작은 표준편차 특징 제거

# fit_transform 전에 불필요한 메모리 해제
gc.collect()

print("\n--- autofeat 특징 생성 및 선택 시작 ---")
X_train_autofeat = af.fit_transform(X_train, y_train)

print("\n--- autofeat 특징 분석 결과 ---")
print("autofeat를 통해 선택된 중요 특징 개수:", X_train_autofeat.shape[1])
print("autofeat를 통해 선택된 중요 특징 목록:")
for feature in af.feature_names_:
    print(feature)

print("\n변환된 학습 데이터의 상위 5개 행:")
print(X_train_autofeat.head())

# X_train은 더 이상 필요 없으므로 메모리에서 해제
del X_train, y_train
gc.collect()
print(f"X_train 해제 후 X_train_autofeat 메모리 사용량: {X_train_autofeat.memory_usage(deep=True).sum() / (1024**2):.2f} MB")


# --- 테스트 데이터 변환 ---
# 학습된 autofeat 모델을 사용하여 테스트 데이터에도 동일한 특징 변환을 적용합니다.
# transform 전에 불필요한 메모리 해제
gc.collect()
X_test_autofeat = af.transform(X_test)
print("\n변환된 테스트 데이터의 상위 5개 행:")
print(X_test_autofeat.head())

# X_test, y_test는 이제 autofeat 변환이 완료되었으므로 해제
del X_test, y_test
gc.collect()
print(f"X_test 해제 후 X_test_autofeat 메모리 사용량: {X_test_autofeat.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

print("\n모든 작업 완료.")