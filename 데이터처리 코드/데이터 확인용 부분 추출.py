import pandas as pd

# 원본 CSV 파일 불러오기
#df = pd.read_csv('../raw/transaction_joined.csv', dtype={'zip': str})
df = pd.read_csv('../raw/transaction_joined.csv')

# 상위 300개 행 추출
data_part = df.head(10000)

# 새로운 CSV 파일로 저장
data_part.to_csv('../raw/data_part.csv', index=False)
