import pandas as pd

client_df = pd.read_csv("../raw/users_data.csv")
card_df = pd.read_csv("../raw/cards_data.csv")
trans_df = pd.read_csv("../raw/transactions_fraud_label_data.csv")

# =========================
# 전처리 함수
# =========================

def clean_money(col):
    """금액 문자열을 float으로 변환 (괄호 포함해도 양수로 통일)"""
    return col.replace(r'[\$,()]', '', regex=True).astype(float)

# 금액 컬럼 정제
client_df["per_capita_income"] = clean_money(client_df["per_capita_income"])
client_df["yearly_income"] = clean_money(client_df["yearly_income"])
client_df["total_debt"] = clean_money(client_df["total_debt"])
card_df["credit_limit"] = clean_money(card_df["credit_limit"])
trans_df["amount"] = clean_money(trans_df["amount"])

# 날짜 변환
trans_df["date"] = pd.to_datetime(trans_df["date"], errors="coerce")
card_df["acct_open_date"] = pd.to_datetime(card_df["acct_open_date"], format="%m/%Y", errors="coerce")

# 불필요한 컬럼 제거
client_df = client_df.drop(columns=["birth_year", "birth_month", "address"])
card_df = card_df.drop(columns=["card_number", "cvv", "expires"])

# Join (거래 기준)
merged_df = trans_df.merge(client_df, left_on="client_id", right_on="id", suffixes=("", "_client"))
merged_df = merged_df.merge(card_df, left_on="card_id", right_on="id", suffixes=("", "_card"))
merged_df = merged_df.drop(columns=["id_client", "id_card"])  # 중복된 고객/카드 id 제거

# 저장
merged_df.to_csv("../raw/transaction_joined.csv", index=False)
