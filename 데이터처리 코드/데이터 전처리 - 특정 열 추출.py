import pandas as pd

def extract_and_save_errors_to_csv(input_file_path, output_file_path, col_name):
    """
    CSV 파일에서 'errors' 속성 중 값이 있는 경우 해당 값들을 추출해 새로운 CSV 파일로 저장

    Args:
        input_file_path (str): 원본 CSV 파일 경로
        output_file_path (str): 추출된 오류 값들을 저장할 CSV 파일 경로
        col_name (str): 추출할 컬럼의 이름
    """
    try:
        # CSV 파일을 데이터프레임으로 읽기
        df = pd.read_csv(input_file_path)

        # 추출할 컬럼이 데이터프레임에 있는지 확인
        if col_name not in df.columns:
            print(f"오류: '{input_file_path}' 파일에 '{col_name}' 컬럼이 존재하지 않습니다.")
            return

        # 추출할 컬럼에서 NaN(결측치)이 아닌 값들만 선택
        col_series = df[col_name].dropna()
        
        # 빈 문자열(empty string) 제거 (문자열 타입인 경우)
        if pd.api.types.is_string_dtype(col_series):
            col_series = col_series[col_series != '']
        
        # 추출된 값들을 리스트로 변환
        extracted_col_values = col_series.tolist()

        if extracted_col_values:
            # 추출된 오류 값들을 포함하는 새로운 데이터프레임 생성
            # 컬럼명은 col_name + '_type'으로 지정합니다.
            new_col_name = col_name + '_type'
            output_df = pd.DataFrame(extracted_col_values, columns=[new_col_name])
            
            # 새로운 CSV 파일로 저장
            output_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f"'{len(extracted_col_values)}'개의 오류 값이 '{output_file_path}' 파일에 성공적으로 저장되었습니다.")
        else:
            print("추출할 오류 값이 없으므로 CSV 파일을 생성하지 않습니다.")

    except FileNotFoundError:
        print(f"오류: 지정된 입력 파일이 없습니다: {input_file_path}")
    except pd.errors.EmptyDataError:
        print(f"오류: '{input_file_path}' 파일이 비어 있습니다.")
    except Exception as e:
        print(f"파일 처리 중 오류가 발생했습니다: {e}")


# 원본 CSV 파일의 경로
input_csv_path = '../transactions_data.csv'

# 추출할 컬럼의 이름
col_name = 'errors'

# 추출된 오류 값을 저장할 새 CSV 파일의 경로
output_csv_path = '../extracted_errors.csv'

extract_and_save_errors_to_csv(input_csv_path, output_csv_path, col_name)
