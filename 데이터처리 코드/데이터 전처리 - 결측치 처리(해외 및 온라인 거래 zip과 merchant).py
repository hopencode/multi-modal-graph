import pandas as pd
import csv


def preprocess_zip_and_save_csv(file_path, output_file_path=None):
    """
    CSV 파일을 읽어 'merchant_city'가 'ONLINE'인 경우 및
    미국 외 국가 결제에 대한 'zip' 속성을 전처리
    모든 'zip' 컬럼 값을 5자리 문자열로 통일하여 CSV로 저장

    Args:
        file_path (str): 원본 CSV 파일의 경로
        output_file_path (str, optional): 전처리된 데이터를 저장할 CSV 파일 경로
                                          None이면 결과를 출력만 합니다. 기본값은 None

    Returns:
        pandas.DataFrame: 전처리된 데이터프레임
    """
    try:
        df = pd.read_csv(file_path, dtype={'zip': str, 'merchant_state': str})

        # 데이터 로드 후, 빈 문자열이나 'NULL' 등이 여전히 남아있을 경우 명시적으로 pd.NA로 변환
        df.replace({'zip': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)
        df.replace({'merchant_state': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)

        print(f"--- '{file_path}' 파일 로드 완료 ---")
        df.info()
        print(f"초기 'zip' 컬럼의 데이터 타입: {df['zip'].dtype}")

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
        return None
    except pd.errors.EmptyDataError:
        print(f"오류: '{file_path}' 파일이 비어 있습니다.")
        return None
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

    # --- 1. 'merchant_city'가 'ONLINE'인 경우 처리 ---
    online_merchants_condition = df['merchant_city'] == 'ONLINE'
    df.loc[online_merchants_condition & df['merchant_state'].isna(), 'merchant_state'] = 'ONLINE'
    if 'zip' in df.columns:
        df.loc[online_merchants_condition & df['zip'].isna(), 'zip'] = '00000'
    print("\n'ONLINE' 결제에 대한 전처리를 완료했습니다.")

    # --- 2. 미국 주/영토 목록 정의 ---
    us_states_and_territories = [
        'AA', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
        'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI',
        'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',
        'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT',
        'WA', 'WI', 'WV', 'WY', 'PR', 'VI', 'GU', 'AS', 'MP'
    ]

    # --- 3. 해외 국가별 ZIP 코드 처리 ---
    # 'merchant_state'가 비어있지 않고, 미국 주/영토가 아니며, 'ONLINE'도 아닌 경우를 '해외'로 간주
    international_condition = (df['merchant_state'].notna()) & \
                              (~df['merchant_state'].isin(us_states_and_territories)) & \
                              (df['merchant_city'] != 'ONLINE')

    if 'zip' in df.columns:
        foreign_zip_missing_condition = international_condition & df['zip'].isna()

        # 해외 국가별 더미 ZIP 코드 매핑 딕셔너리
        # 필요에 따라 이 딕셔너리를 더 확장하세요.
        country_zip_map = {
            'MEXICO': 'MEX00', 'VATICAN CITY': 'VAT00', 'DOMINICAN REPUBLIC': 'DOM00',
            'CANADA': 'CAN00', 'ALBANIA': 'ALB00', 'ALGERIA': 'ALG00', 'ANDORRA': 'AND00',
            'ARGENTINA': 'ARG00', 'ARUBA': 'ARU00', 'AUSTRALIA': 'AUS00',
            'AUSTRIA': 'AUT00', 'AZERBAIJAN': 'AZE00', 'BAHRAIN': 'BAH00',
            'BANGLADESH': 'BGD00', 'BARBADOS': 'BRB00', 'BELGIUM': 'BEL00',
            'BELIZE': 'BLZ00', 'BENIN': 'BEN00', 'BOSNIA AND HERZEGOVINA': 'BIH00',
            'BRAZIL': 'BRA00', 'BRUNEI': 'BRN00', 'BURKINA FASO': 'BFA00',
            'CABO VERDE': 'CPV00', 'CAMEROON': 'CMR00', 'CHILE': 'CHL00',
            'CHINA': 'CHN00', 'COLOMBIA': 'COL00', 'COSTA RICA': 'CRI00',
            'COTE D IVOIRE': 'CIV00', 'CROATIA': 'HRV00', 'CYPRUS': 'CYP00',
            'CZECH REPUBLIC': 'CZE00', 'DENMARK': 'DNK00', 'EAST TIMOR (TIMOR-LESTE)': 'TLS00',
            'ECUADOR': 'ECU00', 'EGYPT': 'EGY00', 'EQUATORIAL GUINEA': 'GNQ00',
            'ERITREA': 'ERI00', 'ESTONIA': 'EST00', 'ETHIOPIA': 'ETH00',
            'FIJI': 'FJI00', 'FINLAND': 'FIN00', 'FRANCE': 'FRA00',
            'GABON': 'GAB00', 'GEORGIA': 'GEO00', 'GERMANY': 'DEU00',
            'GHANA': 'GHA00', 'GREECE': 'GRC00', 'GUATEMALA': 'GTM00',
            'GUINEA': 'GIN00', 'GUYANA': 'GUY00', 'HAITI': 'HTI00',
            'HONDURAS': 'HND00', 'HONG KONG': 'HKG00', 'HUNGARY': 'HUN00',
            'ICELAND': 'ISL00', 'INDIA': 'IND00', 'INDONESIA': 'IDN00',
            'IRAN': 'IRN00', 'IRAQ': 'IRQ00', 'IRELAND': 'IRL00',
            'ISRAEL': 'ISR00', 'ITALY': 'ITA00', 'JAMAICA': 'JAM00',
            'JAPAN': 'JPN00', 'JORDAN': 'JOR00', 'KENYA': 'KEN00',
            'KOSOVO': 'KOS00', 'KYRGYZSTAN': 'KGZ00', 'LATVIA': 'LVA00',
            'LEBANON': 'LBN00', 'LIBERIA': 'LBR00', 'LITHUANIA': 'LTU00',
            'LUXEMBOURG': 'LUX00', 'MACEDONIA': 'MKD00', 'MALAYSIA': 'MYS00',
            'MALDIVES': 'MDV00', 'MALI': 'MLI00', 'MALTA': 'MLT00',
            'MARSHALL ISLANDS': 'MHL00', 'MICRONESIA': 'FSM00', 'MOLDOVA': 'MDA00',
            'MONACO': 'MCO00', 'MONGOLIA': 'MNG00', 'MONTENEGRO': 'MNE00',
            'MOROCCO': 'MAR00', 'MOZAMBIQUE': 'MOZ00', 'MYANMAR (BURMA)': 'MMR00',
            'NAURU': 'NRU00', 'NETHERLANDS': 'NLD00', 'NEW ZEALAND': 'NZL00',
            'NIGER': 'NER00', 'NIGERIA': 'NGA00', 'NORWAY': 'NOR00',
            'OMAN': 'OMN00', 'PAKISTAN': 'PAK00', 'PANAMA': 'PAN00',
            'PAPUA NEW GUINEA': 'PNG00', 'PERU': 'PER00', 'PHILIPPINES': 'PHL00',
            'POLAND': 'POL00', 'PORTUGAL': 'PRT00', 'QATAR': 'QAT00',
            'REPUBLIC OF THE CONGO': 'COG00', 'ROMANIA': 'ROU00', 'RUSSIA': 'RUS00',
            'SAINT VINCENT AND THE GRENADINES': 'VCT00', 'SAMOA': 'WSM00',
            'SAUDI ARABIA': 'SAU00', 'SENEGAL': 'SEN00', 'SERBIA': 'SRB00',
            'SEYCHELLES': 'SYC00', 'SIERRA LEONE': 'SLE00', 'SINGAPORE': 'SGP00',
            'SLOVAKIA': 'SVK00', 'SLOVENIA': 'SVN00', 'SOLOMON ISLANDS': 'SLB00',
            'SOUTH AFRICA': 'ZAF00', 'SOUTH KOREA': 'KOR00', 'SOUTH SUDAN': 'SSD00',
            'SPAIN': 'ESP00', 'SRI LANKA': 'LKA00', 'SUDAN': 'SDN00',
            'SURINAME': 'SUR00', 'SWAZILAND': 'SWZ00', 'SWEDEN': 'SWE00',
            'SWITZERLAND': 'CHE00', 'TAIWAN': 'TWN00', 'THAILAND': 'THA00',
            'THE BAHAMAS': 'BHS00', 'TONGA': 'TON00', 'TRINIDAD AND TOBAGO': 'TTO00',
            'TUNISIA': 'TUN00', 'TURKEY': 'TUR00', 'TUVALU': 'TUV00',
            'UKRAINE': 'UKR00', 'UNITED ARAB EMIRATES': 'ARE00', 'UNITED KINGDOM': 'GBR00',
            'URUGUAY': 'URY00', 'UZBEKISTAN': 'UZB00', 'VANUATU': 'VUT00',
            'VENEZUELA': 'VEN00', 'VIETNAM': 'VNM00', 'YEMEN': 'YEM00',
            'ZAMBIA': 'ZMB00', 'ZIMBABWE': 'ZWE00'
        }

        def assign_foreign_zip(row):
            if pd.isna(row['zip']) and foreign_zip_missing_condition[row.name]:
                # merchant_state를 대문자로 변환하여 매핑 키로 사용
                state_upper = str(row['merchant_state']).upper().strip()  # 공백 제거 추가
                return country_zip_map.get(state_upper, 'OTH00')  # 매핑되지 않은 경우 'OTH00' (Other Foreign)
            return row['zip']  # 조건에 해당하지 않으면 기존 zip 값 유지

        df['zip'] = df.apply(assign_foreign_zip, axis=1)
        print("\n해외 결제에 대한 빈 'zip' 코드를 국가별 더미 값으로 채웠습니다.")
    else:
        print("경고: 'zip' 컬럼이 데이터프레임에 존재하지 않아 'zip' 전처리를 건너뜁니다.")

    # --- 4. 모든 'zip' 값을 5자리 문자열로 표준화 ---
    if 'zip' in df.columns:
        def format_zip(zip_val):
            if pd.isna(zip_val):
                return pd.NA
            str_val = str(zip_val).strip()  # 공백 제거

            try:
                # 숫자처럼 보이는 값만 변환 (예: '74837.0', '123', '00123')
                # 숫자로 변환 가능한지 확인하고, 정수로 변환 후 5자리로 채움
                numeric_val = float(str_val)
                # 정수형으로 변환 후 5자리로 채움
                return str(int(numeric_val)).zfill(5)
            except ValueError:
                # 숫자로 변환 불가능한 경우 (예: 'MEX00', 'OTH00', '100-0001')
                # 해당 문자열 그대로 반환
                return str_val

        df['zip'] = df['zip'].apply(format_zip)
        print("\n모든 'zip' 컬럼 값을 5자리 문자열로 표준화했습니다.")
    else:
        print("경고: 'zip' 컬럼이 데이터프레임에 존재하지 않아 'zip' 전처리를 건너뜁니다.")

    print("\n--- 전처리 완료된 데이터프레임 정보 ---")
    df.info()
    print("\n--- 'zip' 컬럼의 최종 데이터 타입 ---")
    print(df['zip'].dtype)

    # --- 5. 전처리된 데이터를 새로운 CSV 파일로 저장 ---
    if output_file_path:
        try:
            # QUOTE_NONNUMERIC은 숫자처럼 보이는 필드를 따옴표로 묶어 문자열로 강제
            # 이 설정이 메모장에서 '00000'이 '0.0'으로 보이는 현상을 막는 데 가장 중요
            df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"\n전처리된 데이터가 '{output_file_path}'에 성공적으로 저장되었습니다.")

            # 저장 직후, 저장된 파일을 다시 읽어 'zip' 컬럼이 제대로 저장되었는지 확인
            print(f"\n--- '{output_file_path}' 파일을 다시 읽어 'zip' 컬럼 확인 ---")
            re_read_df = pd.read_csv(output_file_path, dtype={'zip': str})
            print(re_read_df[['merchant_city', 'merchant_state', 'zip']].head(20))
            print(f"다시 읽은 'zip' 컬럼의 타입: {re_read_df['zip'].dtype}")

        except Exception as e:
            print(f"전처리된 데이터를 저장하는 중 오류가 발생했습니다: {e}")

    return df

if __name__ == "__main__":

    input_csv_file = '../raw/transactions_fraud_label_data.csv'
    output_csv_file = '../raw/transactions_fraud_label_data_preprocess.csv'

    processed_df = preprocess_zip_and_save_csv(input_csv_file, output_csv_file)

    if processed_df is not None:
        print("\n--- 최종 전처리된 데이터프레임 미리보기 (저장된 파일 내용과 비교) ---")
        print(processed_df[['merchant_city', 'merchant_state', 'zip']].head(20))