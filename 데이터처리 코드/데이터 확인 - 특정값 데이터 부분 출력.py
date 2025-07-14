import pandas as pd

def extract_rows_by_column_value(input_file_path: str, column_name: str, target_value: str, output_file_path: str):
    """
    CSV 파일에서 특정 컬럼의 값이 설정한 값과 일치하는 행만 추출하여 새로운 CSV 파일로 저장

    Args:
        input_file_path (str): 원본 CSV 파일 경로
        column_name (str): 조건을 적용할 컬럼의 이름
        target_value (str): 컬럼에서 찾을 특정 값
        output_file_path (str): 추출된 행들을 저장할 새로운 CSV 파일 경로
    """
    try:
        df = pd.read_csv(input_file_path)
        print(f"--- '{input_file_path}' 파일 로드 완료. (총 {df.shape[0]} 행, {df.shape[1]} 열) ---")

        if column_name not in df.columns:
            print(f"오류: '{input_file_path}' 파일에 '{column_name}' 컬럼이 존재하지 않습니다. 작업을 중단합니다.")
            return

        # 특정 컬럼의 값이 target_value와 일치하는 행만 필터링
        filtered_df = df[df[column_name].astype(str) == target_value]

        if filtered_df.empty:
            print(f"'{column_name}' 컬럼에서 '{target_value}' 값을 가진 행이 없습니다. 출력 파일을 생성하지 않습니다.")
            return

        filtered_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        print(f"\n'{column_name}' 컬럼에서 '{target_value}' 값을 가진 {len(filtered_df)}개 행이")
        print(f"'{output_file_path}' 파일에 성공적으로 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: 지정된 입력 파일이 없습니다: {input_file_path}")
    except pd.errors.EmptyDataError:
        print(f"오류: '{input_file_path}' 파일이 비어 있습니다.")
    except Exception as e:
        print(f"파일 처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 1. 원본 입력 CSV 파일 경로
    input_csv_file = '../raw/transaction_joined.csv'

    # 2. 조건을 적용할 컬럼 이름
    target_column_name = 'fraud' # 컬럼 이름

    # 3. 컬럼에서 찾을 특정 값 (문자열 형태로 입력)
    value_to_match = '1.0' # 조건 값

    # 4. 필터링된 데이터를 저장할 출력 CSV 파일 경로
    output_csv_file = '../raw/filtered_output_data.csv' # 원하는 출력 파일 경로로 변경

    # 함수 호출
    extract_rows_by_column_value(input_csv_file, target_column_name, value_to_match, output_csv_file)