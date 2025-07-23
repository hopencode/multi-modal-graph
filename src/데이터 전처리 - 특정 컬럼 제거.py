import pandas as pd

def delete_columns(input_file, output_file, columns_to_delete):
    """
    CSV 파일에서 지정된 컬럼들을 삭제
    
    Args:
        input_file (str): 입력 CSV 파일 경로
        output_file (str): 출력 CSV 파일 경로
        columns_to_delete (str or list): 삭제할 컬럼명 (문자열) 또는 컬럼명 리스트
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(input_file)
        
        # 문자열로 들어온 경우 리스트로 변환
        if isinstance(columns_to_delete, str):
            columns_to_delete = [columns_to_delete]
        
        # 존재하는 컬럼만 필터링
        existing_columns = [col for col in columns_to_delete if col in df.columns]
        non_existing_columns = [col for col in columns_to_delete if col not in df.columns]
        
        if non_existing_columns:
            print(f"경고: 다음 컬럼들이 파일에 존재하지 않습니다: {non_existing_columns}")
        
        # 컬럼 삭제
        if existing_columns:
            df = df.drop(columns=existing_columns)
            print(f"삭제된 컬럼: {existing_columns}")
        else:
            print("삭제할 컬럼이 없습니다.")
        
        # 수정된 데이터를 새 파일로 저장
        df.to_csv(output_file, index=False)
        print(f"결과가 {output_file}에 저장되었습니다.")
        
        return df
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "../raw/transaction_joined.csv"
    output_file = "../raw/transaction_joined_select_column.csv"
    
    # 하나의 컬럼 삭제
    #delete_columns(input_file, output_file, "삭제할컬럼")
    
    # 여러 컬럼 삭제
    delete_columns(input_file, output_file, ["client_id_card", "card_on_dark_web"])
