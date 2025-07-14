import pandas as pd

client_df = pd.read_csv("../raw/users_data.csv")
card_df = pd.read_csv("../raw/cards_data.csv")
trans_df = pd.read_csv("../raw/transactions_fraud_label_data_preprocess.csv")

def clean_money(col):
    """금액 문자열을 float으로 변환"""
    if not pd.api.types.is_string_dtype(col):
        return col

    # 금액 문자열에서 달러 기호($)와 쉼표(,)를 제거하고, 앞뒤 공백을 제거
    cleaned_col = col.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()

    # 숫자로 변환할 수 없는 값은 NaN으로 처리하여 오류 없이 변환
    return pd.to_numeric(cleaned_col, errors='coerce')

# 금액 컬럼 정제
client_df["per_capita_income"] = clean_money(client_df["per_capita_income"])
client_df["yearly_income"] = clean_money(client_df["yearly_income"])
client_df["total_debt"] = clean_money(client_df["total_debt"])
card_df["credit_limit"] = clean_money(card_df["credit_limit"])
trans_df["amount"] = clean_money(trans_df["amount"])
trans_df["amount"] = trans_df["amount"].abs()

# 날짜 변환
trans_df["date"] = pd.to_datetime(trans_df["date"], errors="coerce")
card_df["acct_open_date"] = pd.to_datetime(card_df["acct_open_date"], format="%m/%Y", errors="coerce")

# 불필요한 컬럼 제거
# client_df = client_df.drop(columns=["birth_year", "birth_month", "address"])
# card_on_dark_web 속성은 전부 No임
card_df = card_df.drop(columns=["card_number", "cvv", "card_on_dark_web"])

# Join (거래 기준)
merged_df = trans_df.merge(client_df, left_on="client_id", right_on="id", suffixes=("", "_client"))
merged_df = merged_df.merge(card_df, left_on="card_id", right_on="id", suffixes=("", "_card"))
merged_df = merged_df.drop(columns=["id_client", "id_card", "client_id_card"])  # 중복된 고객/카드 id 제거, 카드 주인의 id 제거

# 저장
merged_df.to_csv("../raw/transaction_joined.csv", index=False)
