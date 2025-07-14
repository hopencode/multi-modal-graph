import pandas as pd

def delete_columns_from_csv(input_filepath, output_filepath, columns_to_delete):
    """
    CSV 파일에서 특정 열을 삭제하고 새 파일로 저장

    Args:
        input_filepath (str): 입력 CSV 파일 경로
        output_filepath (str): 결과 CSV 파일을 저장할 경로
        columns_to_delete (list): 삭제할 열 이름(문자열) 리스트
    """
    try:
        df = pd.read_csv(input_filepath)
        print(f"원본 파일 '{input_filepath}'의 열: {df.columns.tolist()}")

        # 삭제할 열이 데이터프레임에 존재하는지 확인
        existing_columns = [col for col in columns_to_delete if col in df.columns]
        non_existing_columns = [col for col in columns_to_delete if col not in df.columns]

        if non_existing_columns:
            print(f"경고: 다음 열은 원본 파일에 존재하지 않아 삭제할 수 없습니다: {non_existing_columns}")

        if existing_columns:
            df_cleaned = df.drop(columns=existing_columns)
            df_cleaned.to_csv(output_filepath, index=False)
            print(f"'{existing_columns}' 열이 삭제된 파일이 '{output_filepath}' (으)로 성공적으로 저장되었습니다.")
            print(f"새 파일의 열: {df_cleaned.columns.tolist()}")
        else:
            print("삭제할 유효한 열이 없어 새 파일이 원본과 동일하게 저장됩니다.")
            df.to_csv(output_filepath, index=False)


    except FileNotFoundError:
        print(f"오류: '{input_filepath}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

input_file = '../raw/transaction_joined.csv'
output_file = '../raw/transaction_joined_column_selected.csv'
columns_to_remove = ['card_on_dark_web'] # 삭제하고 싶은 열 이름을 리스트로 입력

delete_columns_from_csv(input_file, output_file, columns_to_remove)