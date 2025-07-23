import pandas as pd
import json

def prepare_transaction_data(transactions_file, fraud_labels_file, mcc_codes_file, output_file):
    """
    거래 데이터에 상점 유형(mcc_type)과 이상 거래 라벨(fraud) 컬럼을 추가해 새로운 CSV 파일로 저장
    fraud 라벨링 시, sorted_fraud.csv에 없는 거래는 NaN으로 처리

    Args:
        transactions_file (str): 원본 거래 데이터 CSV 파일 경로
        fraud_labels_file (str): 사기 거래 라벨 CSV 파일 경로
        mcc_codes_file (str): MCC 코드 매핑 JSON 파일 경로
        output_file (str): 결과 데이터가 저장될 CSV 파일 경로
    """
    try:
        df_transactions = pd.read_csv(transactions_file, dtype={'zip': str, 'amount': str})
        df_fraud_labels = pd.read_csv(fraud_labels_file)

        with open(mcc_codes_file, 'r', encoding='utf-8') as f:
            mcc_codes = json.load(f)

        print("데이터 로드 완료.")
        print(f"원본 거래 데이터 레코드 수: {len(df_transactions)}")
        print(f"사기 라벨 데이터 레코드 수: {len(df_fraud_labels)}")

        # mcc_type 컬럼 추가
        df_transactions['mcc_type'] = df_transactions['mcc'].astype(str).map(mcc_codes).fillna('Unknown')
        print("'mcc_type' 컬럼 추가 완료.")

        # fraud 컬럼 추가
        status_to_fraud_map = {
            'No': 0,
            'Yes': 1
        }
        df_fraud_labels['fraud_status_mapped'] = df_fraud_labels['Status'].str.strip().map(status_to_fraud_map)
        df_transactions = pd.merge(
            df_transactions,
            df_fraud_labels[['id', 'fraud_status_mapped']],
            on='id',
            how='left'
        )
        df_transactions['fraud'] = df_transactions['fraud_status_mapped']
        df_transactions = df_transactions.drop(columns=['fraud_status_mapped'])
        print("'fraud' 컬럼 추가 및 업데이트 완료 (0: 정상, 1: 이상, NaN: 라벨 없음).")

        # amount 컬럼 "양수 실수"로 변환
        if 'amount' in df_transactions.columns:
            df_transactions['amount'] = (
                df_transactions['amount']
                .replace('[\$,]', '', regex=True)   # '$', ',' 제거
                .astype(float)
                .abs()
            )
            print("'amount' 컬럼 양수(float)로 변환 완료.")

        df_transactions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"처리된 데이터가 '{output_file}'에 성공적으로 저장되었습니다.")
        print(f"최종 데이터 레코드 수: {len(df_transactions)}")
        print(f"정상 거래 수 (fraud=0): {df_transactions['fraud'].value_counts(dropna=False).get(0.0, 0)}")
        print(f"이상 거래 수 (fraud=1): {df_transactions['fraud'].value_counts(dropna=False).get(1.0, 0)}")
        print(f"라벨 없는 거래 수 (fraud=NaN): {df_transactions['fraud'].isna().sum()}")

    except FileNotFoundError as e:
        print(f"오류: 파일이 없습니다 - {e}")
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파일 파싱 중 문제가 발생했습니다 - {e}")
    except KeyError as e:
        print(f"오류: 필요한 컬럼이 데이터에 없습니다 - {e}")
    except Exception as e:
        print(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    transactions_data_path = '../raw/transactions_data.csv'
    sorted_fraud_path = '../raw/sorted_fraud.csv'
    mcc_codes_path = '../raw/mcc_codes.json'
    output_data_path = '../raw/transactions_fraud_label.csv'

    prepare_transaction_data(transactions_data_path, sorted_fraud_path, mcc_codes_path, output_data_path)

    try:
        df_final = pd.read_csv(output_data_path)
        print("\n--- 최종 생성된 파일의 상위 5개 행 ---")
        print(df_final.head())
        print("\n--- fraud 컬럼 값 분포 ---")
        print(df_final['fraud'].value_counts(dropna=False))
        print("\n--- mcc_type 컬럼 값 분포 ---")
        print(df_final['mcc_type'].value_counts().head())
        print("\n--- amount 컬럼 요약 ---")
        print(df_final['amount'].describe())
    except Exception as e:
        print(f"최종 파일 확인 중 오류: {e}")
