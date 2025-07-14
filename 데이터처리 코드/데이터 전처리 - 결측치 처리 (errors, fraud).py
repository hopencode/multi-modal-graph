import pandas as pd
import csv


def clean_errors_and_fraud_data(file_path, output_file_path=None):
    """
    CSV 파일을 읽어 'errors' 컬럼의 결측값을 'No Error'로 채우고,
    'fraud' 컬럼에 결측값이 있는 행을 삭제합니다.
    전처리된 데이터를 새로운 CSV 파일로 저장합니다.

    Args:
        file_path (str): 원본 CSV 파일의 경로.
        output_file_path (str, optional): 전처리된 데이터를 저장할 CSV 파일 경로.
                                          None이면 결과를 출력만 합니다. 기본값은 None.

    Returns:
        pandas.DataFrame: 전처리된 데이터프레임.
    """
    try:
        # CSV 파일을 읽을 때 'errors'와 'fraud' 컬럼을 문자열로 명시적으로 지정합니다.
        # na_values를 사용하여 빈 문자열, 'NULL' 등도 결측값(NaN/pd.NA)으로 인식하게 합니다.
        df = pd.read_csv(file_path, dtype={'errors': str, 'fraud': str})

        # 데이터 로드 후, 'errors'와 'fraud' 컬럼의 빈 문자열 등이 아직 남아있을 경우
        # 명시적으로 pd.NA로 변환하여 .isna()가 정확하게 동작하도록 합니다.
        df.replace({'errors': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)
        df.replace({'fraud': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)

        print(f"--- '{file_path}' 파일 로드 완료 ---")
        df.info()
        print(f"\n로드 후 'errors' 컬럼의 고유값:\n{df['errors'].value_counts(dropna=False)}")
        print(f"\n로드 후 'fraud' 컬럼의 고유값:\n{df['fraud'].value_counts(dropna=False)}")

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
        return None
    except pd.errors.EmptyDataError:
        print(f"오류: '{file_path}' 파일이 비어 있습니다.")
        return None
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

    # --- 1. 'errors' 컬럼 결측값 처리 ---
    # 'errors' 컬럼이 데이터프레임에 있는지 확인
    if 'errors' in df.columns:
        initial_errors_na_count = df['errors'].isna().sum()
        # 결측값을 'No Error' 문자열로 채움
        df['errors'] = df['errors'].fillna('No Error')
        if initial_errors_na_count > 0:
            print(f"\n'errors' 컬럼의 결측값 {initial_errors_na_count}개를 'No Error'로 채웠습니다.")
        else:
            print("\n'errors' 컬럼에 결측값이 없어 채워진 값이 없습니다.")
    else:
        print("경고: 'errors' 컬럼이 데이터프레임에 존재하지 않아 해당 전처리를 건너뜁니다.")

    # --- 2. 'fraud' 컬럼 결측값 행 삭제 ---
    # 'fraud' 컬럼이 데이터프레임에 있는지 확인
    if 'fraud' in df.columns:
        initial_rows = len(df)  # 행 삭제 전의 초기 행 개수를 저장
        print(f"\n결측값 행 삭제 전 초기 전체 행 개수: {initial_rows}")
        # 'fraud' 컬럼에 NaN (결측값)이 있는 모든 행을 삭제
        df.dropna(subset=['fraud'], inplace=True)
        rows_dropped = initial_rows - len(df)  # 삭제된 행의 개수를 계산
        if rows_dropped > 0:
            print(f"\n'fraud' 컬럼에 결측값이 있는 행 {rows_dropped}개를 삭제했습니다.")
            print(f"삭제 후 남은 행 개수: {len(df)}")
        else:
            print("\n'fraud' 컬럼에 결측값이 없어 삭제된 행이 없습니다.")
    else:
        print("경고: 'fraud' 컬럼이 데이터프레임에 존재하지 않아 해당 전처리를 건너뜁니다. (삭제된 행 없음)")

    print("\n--- 전처리 완료된 데이터프레임 정보 ---")
    df.info()
    print("\n--- 'errors' 컬럼 최종 고유값 확인 ---")
    print(df['errors'].value_counts(dropna=False))  # dropna=False를 통해 NaN이 없는지 확인
    print("\n--- 'fraud' 컬럼 최종 고유값 확인 ---")
    print(df['fraud'].value_counts(dropna=False))  # dropna=False를 통해 NaN이 없는지 확인

    # --- 3. 전처리된 데이터를 새로운 CSV 파일로 저장 ---
    if output_file_path:
        try:
            # QUOTE_NONNUMERIC은 숫자처럼 보이는 필드를 따옴표로 묶어 문자열로 강제
            df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"\n전처리된 데이터가 '{output_file_path}'에 성공적으로 저장되었습니다.")

            # 저장 직후, 저장된 파일을 다시 읽어 'errors'와 'fraud' 컬럼이 제대로 처리되었는지 확인
            print(f"\n--- '{output_file_path}' 파일을 다시 읽어 'errors' 및 'fraud' 컬럼 확인 ---")
            re_read_df = pd.read_csv(output_file_path, dtype={'errors': str, 'fraud': str})
            print(f"다시 읽은 'errors' 컬럼 고유값:\n{re_read_df['errors'].value_counts(dropna=False)}")
            print(f"다시 읽은 'fraud' 컬럼 고유값:\n{re_read_df['fraud'].value_counts(dropna=False)}")

        except Exception as e:
            print(f"전처리된 데이터를 저장하는 중 오류가 발생했습니다: {e}")

    return df

if __name__ == "__main__":

    input_csv_file = '../raw/transactions_fraud_label_data_preprocess.csv'
    output_csv_file = '../raw/transactions_fraud_label_data_preprocess.csv'

    cleaned_df = clean_errors_and_fraud_data(input_csv_file, output_csv_file)

    if cleaned_df is not None:
        print("\n--- 최종 전처리된 데이터프레임 미리보기 ---")
        print(cleaned_df.head(10))