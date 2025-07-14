import pandas as pd
import os


def select_and_save_columns_to_new_csv(input_file_path: str, col_names: list, output_file_path: str):
    """
    CSV 파일에서 col_names에 입력한 컬럼들만 선택하여 새로운 CSV 파일로 저장

    Args:
        input_file_path (str): 원본 CSV 파일 경로
        col_names (list): 새로운 CSV 파일에 포함할 컬럼 이름들의 리스트
        output_file_path (str): 선택된 컬럼들을 저장할 새로운 CSV 파일 경로
    """
    try:
        # CSV 파일을 데이터프레임으로 읽기
        df = pd.read_csv(input_file_path)
        print(f"--- '{input_file_path}' 파일 로드 완료. (총 {df.shape[0]} 행, {df.shape[1]} 열) ---")

        # 데이터프레임에 실제로 존재하는 컬럼만 필터링
        existing_cols = [col for col in col_names if col in df.columns]
        non_existing_cols = [col for col in col_names if col not in df.columns]

        if non_existing_cols:
            print(f"경고: 다음 컬럼들은 원본 파일에 존재하지 않아 제외됩니다: {', '.join(non_existing_cols)}")

        if not existing_cols:
            print("오류: 요청하신 컬럼 중 유효한 컬럼이 하나도 없어 출력 파일을 생성할 수 없습니다.")
            return

        # 필요한 컬럼만 선택하여 새로운 데이터프레임 생성
        output_df = df[existing_cols].copy()

        # 새로운 CSV 파일로 저장
        # 한글 깨짐 방지를 위해 encoding='utf-8-sig' 사용 권장
        output_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        print(f"\n'{', '.join(existing_cols)}' 컬럼이 포함된 새로운 파일이 '{output_file_path}'에 성공적으로 저장되었습니다.")
        print(f"저장된 파일의 크기: {output_df.shape[0]} 행, {output_df.shape[1]} 열")

    except FileNotFoundError:
        print(f"오류: 지정된 입력 파일이 없습니다: {input_file_path}")
    except pd.errors.EmptyDataError:
        print(f"오류: '{input_file_path}' 파일이 비어 있습니다.")
    except Exception as e:
        print(f"파일 처리 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # 원본 CSV 파일의 경로
    input_csv_path = '../raw/transactions_fraud_label_data_preprocess.csv'

    # 새로운 CSV 파일에 포함할 컬럼 이름들의 리스트
    columns_to_select = ['amount', 'fraud']

    # 선택된 컬럼들을 저장할 새로운 CSV 파일의 경로
    output_csv_path = '../raw/amout_fraud.csv'

    # 함수 호출
    select_and_save_columns_to_new_csv(input_csv_path, columns_to_select, output_csv_path)
