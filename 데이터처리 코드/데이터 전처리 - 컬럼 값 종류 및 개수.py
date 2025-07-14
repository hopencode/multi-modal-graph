import pandas as pd

def analyze_column_values(filepath, column_name, output_filepath=None):
    """
    CSV 파일의 특정 열에 있는 값의 종류와 각 값의 개수를 출력
    선택적으로 분석 결과를 CSV 파일로 저장

    Args:
        filepath (str): CSV 파일 경로
        column_name (str): 분석할 열의 이름
        output_filepath (str, optional): 분석 결과를 저장할 CSV 파일 경로. 기본값은 None (저장 안 함)
    """
    try:
        df = pd.read_csv(filepath)

        # 분석할 열이 데이터프레임에 존재하는지 확인
        if column_name not in df.columns:
            print(f"오류: '{column_name}' 열은 '{filepath}' 파일에 존재하지 않습니다.")
            print(f"사용 가능한 열: {df.columns.tolist()}")
            return

        # 특정 열의 값 종류와 개수 세기
        # value_counts()는 각 고유 값의 개수를 Series 형태로 반환
        value_counts = df[column_name].value_counts().sort_index()

        print(f"\n--- '{filepath}' 파일의 '{column_name}' 열 값 분석 결과 ---")
        print(f"값이 있는 행 개수: {df[column_name].count()}")
        print(value_counts)

        if output_filepath:
            # 결과를 DataFrame으로 변환하여 CSV로 저장
            # Series의 인덱스는 '값'이 되고, 값은 '개수'가 되도록 설정
            value_counts_df = value_counts.rename('Count').reset_index()
            value_counts_df.columns = ['Value', 'Count'] # 컬럼 이름 재설정
            value_counts_df.to_csv(output_filepath, index=False)
            print(f"분석 결과가 '{output_filepath}' (으)로 성공적으로 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":

    input_file_for_column_analysis = '../raw/transactions_fraud_label_data.csv'
    column_to_analyze = 'merchant_state'
    output_file_for_column_analysis_results = None
    # output_file_for_column_analysis_results = '../raw/merchant_state_data_count.csv' # 분석 결과를 저장할 파일 경로 (선택 사항)

    analyze_column_values(input_file_for_column_analysis, column_to_analyze, output_file_for_column_analysis_results)