import pandas as pd

def join_csv_and_balance():
    try:
        client_df = pd.read_csv("../raw/users_data.csv")
        card_df = pd.read_csv("../raw/cards_data.csv")
        trans_df = pd.read_csv("../raw/transactions_fraud_label_preprocess.csv", dtype={'zip': str})
    except Exception as e:
        print(f"파일 로딩 오류: {e}")
        return None

    print("데이터 로드 완료.\n")

    # 불필요한 컬럼 제거
    # card_on_dark_web 속성은 전부 No임
    card_df = card_df.drop(columns=["card_number", "cvv", "card_on_dark_web"])

    # Join (거래 기준)
    merged_df = trans_df.merge(client_df, left_on="client_id", right_on="id", suffixes=("", "_client"))
    merged_df = merged_df.merge(card_df, left_on="card_id", right_on="id", suffixes=("", "_card"))
    merged_df = merged_df.drop(columns=["id_client", "id_card", "client_id_card"])  # 중복된 고객/카드 id 제거, 카드 주인의 id 제거
    print("거래 데이터 기준 Join 완료\n")

    # 사기 거래 / 정상 거래 분리
    df_fraud = merged_df[merged_df["fraud"] == 1]
    df_notfraud = merged_df[merged_df["fraud"] == 0]
    print("사기 거래 / 정상 거래 분리 완료")

    # 정상 거래 다운샘플링 (사기 거래 수의 10배)
    notfraud_sampled = df_notfraud.sample(n=len(df_fraud) * 10, random_state=42)
    print("정상 거래 다운샘플링 완료")

    # 합치기 (사기 거래 전부 + 다운샘플링된 정상 거래)
    df_balanced = pd.concat([df_fraud, notfraud_sampled])
    df_balanced = df_balanced.sort_values('date').reset_index(drop=True)
    print("추출한 데이터 합치기 완료\n")

    # Join 및 파생속성 생성 후 증강 파일 저장
    df_balanced.to_csv("../raw/transaction_joined_balance.csv", index=False)
    print("파일 저장 완료")


if __name__ == "__main__":
    join_csv_and_balance()