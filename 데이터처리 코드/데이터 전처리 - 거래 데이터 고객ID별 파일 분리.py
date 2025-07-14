import pandas as pd
import os

def split_csv_by_client_id(input_file_path, output_directory=None):
    """
    대용량 거래 CSV 파일을 client_id 속성을 기준으로 분할하여 각 고객별 파일로 저장

    Args:
        input_file_path (str): 분할할 원본 CSV 파일 경로
        output_directory (str, optional): 분할된 파일들을 저장할 디렉토리 경로
                                          지정하지 않으면 원본 파일이 있는 디렉토리 아래에
                                          'client_transactions' 폴더 생성
    """
    try:
        print(f"'{input_file_path}' 파일을 로드 중입니다...")
        df = pd.read_csv(input_file_path)
        print(f"파일 로드 완료. 총 {len(df)}개의 레코드가 있습니다.")

        # 출력 디렉토리 미입력 시 입력 데이터와 같은 위치로 설정
        if output_directory is None:
            base_dir = os.path.dirname(input_file_path)
            output_directory = os.path.join(base_dir, 'client_transactions')
        
        # 출력 디렉토리 없으면 생성
        os.makedirs(output_directory, exist_ok=True)
        print(f"고객별 파일을 '{output_directory}'에 저장합니다.")

        # client_id 기준으로 그룹화
        unique_client_ids = df['client_id'].unique()
        print(f"총 {len(unique_client_ids)}명의 고유 client_id를 찾았습니다.")

        # 각 client_id별로 데이터프레임을 필터링하여 파일로 저장
        for i, client_id in enumerate(unique_client_ids):
            # 현재 client_id에 해당하는 모든 거래 내역 필터링
            client_df = df[df['client_id'] == client_id]

            output_file_name = f"client_{client_id}_transactions.csv"
            output_file_path = os.path.join(output_directory, output_file_name)

            client_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

            # 진행 상황 표시
            if (i + 1) % 100 == 0 or (i + 1) == len(unique_client_ids):
                print(f"진행: {i+1}/{len(unique_client_ids)} - 'client_{client_id}_transactions.csv' 저장 완료 ({len(client_df)} 레코드)")

        print("모든 고객별 파일 분할 및 저장이 완료되었습니다.")

    except FileNotFoundError:
        print(f"오류: 지정된 파일이 없습니다: {input_file_path}")
    except pd.errors.EmptyDataError:
        print(f"오류: '{input_file_path}' 파일이 비어 있습니다.")
    except Exception as e:
        print(f"파일 분할 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # 분할할 원본 파일 경로
    input_combined_csv = '../raw/transactions_fraud_label_data.csv'

    # 고객별 파일들을 저장할 디렉토리 (선택 사항)
    # 지정하지 않으면 input_combined_csv 파일과 같은 위치에 'client_transactions' 폴더 생성
    output_base_dir = '../raw/transactions_data_label_split'

    split_csv_by_client_id(input_combined_csv, output_base_dir)
