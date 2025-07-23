import pandas as pd

def analyze_csv_missing_values(file_path):
    """
    CSV 파일을 읽어 빈 값(NaN)을 분석하고 결과 출력

    Args:
        file_path (str): 분석할 CSV 파일의 경로

    Returns:
        None
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
        return
    except pd.errors.EmptyDataError:
        print(f"오류: '{file_path}' 파일이 비어 있습니다.")
        return
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    print(f"--- '{file_path}' 파일 빈 값 분석 결과 ---")
    print("\n## 1. 각 열의 빈 값 개수:")
    missing_values_per_column = df.isnull().sum()
    print(missing_values_per_column[missing_values_per_column > 0])

    print("\n## 2. 전체 빈 값이 있는 열:")
    columns_with_all_missing = []
    for col in df.columns:
        if df[col].isnull().all():
            columns_with_all_missing.append(col)

    if columns_with_all_missing:
        print(f"  - 전체 빈 값인 열: {', '.join(columns_with_all_missing)}")
    else:
        print("  - 전체 빈 값인 열이 없습니다.")

    print("\n## 3. 전체 빈 값 개수:")
    total_missing_values = df.isnull().sum().sum()
    print(f"  - 파일 전체의 빈 값 개수: {total_missing_values}개")

if __name__ == "__main__":
    csv_file_path = '../raw/transactions_data.csv'

    analyze_csv_missing_values(csv_file_path)