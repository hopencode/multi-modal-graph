import pandas as pd

# 1. 합쳐진 전체 데이터 로드
df = pd.read_csv("../raw/transaction_joined.csv")

# 2. 라벨링된 데이터만 필터링
df_labeled = df[df['fraud'].notna()].copy()
'''labeled_path = "./transaction_labeled.csv"
df_labeled.to_csv(labeled_path, index=False)'''

# 3. 사기 거래 / 정상 거래 분리
df_fraud = df_labeled[df_labeled["fraud"] == 1]
df_nonfraud = df_labeled[df_labeled["fraud"] == 0]

# 4. 정상 거래 다운샘플링 (사기 거래 수의 10배)
nonfraud_sampled = df_nonfraud.sample(n=len(df_fraud) * 10, random_state=42)

# 5. 합치기 (사기 거래 전부 + 다운샘플링된 정상 거래)
df_balanced = pd.concat([df_fraud, nonfraud_sampled]).sample(frac=1, random_state=42)  # 섞기
balanced_path = "../raw/transaction_labeled_balanced.csv"
df_balanced.to_csv(balanced_path, index=False)
