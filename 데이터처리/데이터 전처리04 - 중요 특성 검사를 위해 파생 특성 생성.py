import pandas as pd

def make_extra_features():
    try:
        df = pd.read_csv("../raw/transaction_joined_balance.csv", dtype={'zip': str})
    except Exception as e:
        print(f"파일 로딩 오류: {e}")
        return None
    print("데이터 로드 완료.\n")

    def clean_money(col):
        """금액 문자열을 float으로 변환"""
        if not pd.api.types.is_string_dtype(col):
            return col

        # 금액 문자열에서 달러 기호($)와 쉼표(,)를 제거하고, 앞뒤 공백을 제거
        cleaned_col = col.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()

        # 숫자로 변환할 수 없는 값은 NaN으로 처리하여 오류 없이 변환
        return pd.to_numeric(cleaned_col, errors='coerce')

    # 금액 컬럼 정제
    df["per_capita_income"] = clean_money(df["per_capita_income"])
    df["yearly_income"] = clean_money(df["yearly_income"])
    df["total_debt"] = clean_money(df["total_debt"])
    df["credit_limit"] = clean_money(df["credit_limit"])
    df["amount"] = clean_money(df["amount"])
    df["amount"] = df["amount"].abs()
    print("금액 컬럼 정제 완료")

    # 날짜 변환
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["acct_open_date"] = pd.to_datetime(df["acct_open_date"], format="%m/%Y", errors="coerce")
    print("날짜 컬럼 변환 완료")

    # 카드 만료일을 해당 월의 말일로 변환
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

    df['expires_last_day'] = df['expires'].apply(convert_expires_to_last_day)

    # 파생 변수 생성
    df['transaction_hour'] = df['date'].dt.hour
    df['transaction_dayofweek'] = df['date'].dt.dayofweek
    df['account_age_days'] = (df['date'] - df['acct_open_date']).dt.days
    df['months_to_expiry'] = (df['expires_last_day'].dt.year - df['date'].dt.year) * 12 + (
                df['expires_last_day'].dt.month - df['date'].dt.month)
    df['transaction_age'] = df['date'].dt.year - df['birth_year']
    df['years_to_retirement'] = df['retirement_age'] - df['transaction_age']
    df['zip_prefix'] = df['zip'].astype(str).str[:3]
    print("파생 속성 생성 완료")

    # 불필요/중복/식별자 컬럼 삭제
    drop_cols = [
        'id', 'card_id',
        # 'client_id', 'merchant_id',
        'expires', 'acct_open_date', 'birth_year', 'birth_month',
        'current_age', 'retirement_age', 'mcc_type',
        'zip', 'date'
    ]
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols)
    print("불필요/중복/식별자 컬럼 삭제 완료\n")

    df.to_csv("../raw/transaction_joined_balance_feature_preprocess.csv", index=False)
    print("파일 저장 완료")
    
if __name__ == "__main__":
    make_extra_features()