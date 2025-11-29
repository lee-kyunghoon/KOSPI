import yfinance as yf
import pandas as pd

"""
1. 시간대 차이 처리
문제: 미국 시장 종가(미국 동부시간 16:00) = 한국시간 다음날 새벽 5~6시
해결: 미국 데이터를 1일 shift하여 한국 투자자 관점에 맞춤
      ex)한국 10/2일 데이터 = 미국 10/1일 종가 사용
        미국 데이터 다운로드 시 시작 날짜를 하루 앞당김 (09/30부터 다운로드)

2. 휴장일 처리
한국 휴장일 (추석, 설날 등):
    KOSPI를 기준 인덱스로 사용
    한국이 쉬는 날은 데이터에 포함 안 됨
미국 휴장일 (추수감사절, 독립기념일 등):
    Forward fill: 가장 최근 거래일 데이터로 채움
    Backward fill: 시작 날짜에 데이터 없을 때 대비
"""

def download_stock_data(ticker, start_date, end_date):
    """
    주어진 티커에 대해 yfinance를 사용하여 주식 데이터를 다운로드합니다.
    
    매개변수:
    ticker (str): 주식 티커 심볼
    start_date (str): 데이터 시작 날짜 (예: '2010-01-01')
    end_date (str): 데이터 종료 날짜 (예: '2025-10-31')
    
    반환값:
    pd.DataFrame: 다운로드된 주식 데이터
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            print(f"⚠️  Warning: '{ticker}' 티커에 대한 데이터를 다운로드할 수 없습니다.")
            return pd.DataFrame()
        
        # 다운로드 직후 NaN 체크
        if data.isna().any().any():
            nan_count = data.isna().sum().sum()
            print(f"⚠️  Warning: 다운로드된 데이터에 {nan_count}개의 NaN 값이 포함되어 있습니다.")
        
        return data
    except Exception as e:
        print(f"❌ Error: 데이터 다운로드 중 오류 발생 - {str(e)}")
        return pd.DataFrame()


# 데이터 정체 (한국 기준으로 미국 휴장일 경우 이전 거래일 데이터 사용)
def fill_missing_dates(data, reference_dates):
    """
    기준 날짜(KOSPI 거래일)에 맞춰 결측된 날짜를 채우고, 결측된 날짜의 데이터를 이전 거래일의 데이터로 채웁니다.
    
    매개변수:
    data (pd.DataFrame): 원본 주식 데이터
    reference_dates (DatetimeIndex): 기준이 되는 날짜 인덱스 (KOSPI 거래일)
    
    반환값:
    pd.DataFrame: 결측된 날짜가 채워진 주식 데이터
    """
    # 기준 날짜로 인덱스 재설정
    data = data.reindex(reference_dates)
    
    # NaN 체크 (원본 데이터 확인)
    if data.isna().any().any():
        print(f"⚠️  Warning: 원본 데이터에 {data.isna().sum().sum()}개의 NaN 값이 있습니다.")
        print(f"NaN이 있는 컬럼: {data.columns[data.isna().any()].tolist()}")
    
    # Forward fill (이전 거래일 데이터로 채우기)
    data = data.ffill()
    
    # 시작 날짜에 데이터가 없는 경우를 위해 Backward fill 추가
    data = data.bfill()
    
    # 여전히 NaN이 남아있는지 확인
    remaining_nans = data.isna().sum().sum()
    if remaining_nans > 0:
        print(f"❌ Error: {remaining_nans}개의 NaN 값이 여전히 남아있습니다.")
        print(f"NaN이 있는 컬럼: {data.columns[data.isna().any()].tolist()}")
        # 최후의 수단: 0으로 채우기 (또는 dropna 사용 가능)
        data = data.fillna(0)
        print("→ 남은 NaN 값을 0으로 채웠습니다.")
    else:
        print("✅ 모든 NaN 값이 성공적으로 채워졌습니다.")
    
    return data

def create_combined_dataset(start_date, end_date):
    """
    여러 티커의 데이터를 다운로드하고 하나의 DataFrame으로 결합합니다.
    KOSPI 거래일을 기준으로 모든 데이터를 정렬합니다.
    
    *** 중요: 시간대 처리 ***
    - 미국 시장 종가(미국시간 16:00) = 한국시간 다음날 새벽 5시/6시
    - 따라서 미국 데이터를 1일 shift하여 한국 거래일에 맞춤
    - 예: 한국 10/2일 → 미국 10/1일 데이터 사용
    
    매개변수:
    start_date (str): 데이터 시작 날짜
    end_date (str): 데이터 종료 날짜
    
    반환값:
    pd.DataFrame: 9개 컬럼을 가진 결합된 데이터
    """
    # 티커 정의 (순서 중요: KOSPI를 먼저 다운로드)
    tickers = {
        'KOSPI': '^KS11',      # 코스피 지수
        'NASDAQ': '^IXIC',     # 나스닥 지수
        'US10Y': '^TNX',       # 미국 10년물 국채 금리
        'VIX': '^VIX',         # VIX 지수
        'USD_KRW': 'KRW=X'     # 원/달러 환율
    }
    
    combined_df = None
    kospi_dates = None  # KOSPI 거래일을 저장할 변수
    
    for name, ticker in tickers.items():
        print(f"\n다운로드 중: {name} ({ticker})")
        
        # 미국 데이터는 시작 날짜를 하루 앞당겨서 다운로드
        # (shift 후 매칭을 위해)
        if name != 'KOSPI':
            adjusted_start = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"📅 시작 날짜 조정: {start_date} → {adjusted_start} (shift 보정)")
            data = download_stock_data(ticker, adjusted_start, end_date)
        else:
            data = download_stock_data(ticker, start_date, end_date)
        
        if not data.empty:
            # KOSPI의 경우: 거래일을 기준으로 설정
            if name == 'KOSPI':
                kospi_dates = data.index  # KOSPI 거래일 저장
                filled_data = data
                print(f"✅ KOSPI 거래일 기준 설정: {len(kospi_dates)}일")
            else:
                # 다른 티커들: 시간대 고려하여 하루 shift 후 KOSPI 거래일에 맞춰 정렬
                if kospi_dates is not None:
                    # 미국 시장 데이터를 1일 뒤로 shift (미국 10/1 → 한국 10/2)
                    data.index = data.index + pd.Timedelta(days=1)
                    print(f"⏰ 시간대 보정: 미국 데이터를 1일 shift")
                    
                    # KOSPI 거래일에 맞춰 정렬
                    filled_data = fill_missing_dates(data, kospi_dates)
                    print(f"✅ KOSPI 거래일에 맞춰 정렬 완료")
                else:
                    print(f"❌ Error: KOSPI 데이터를 먼저 다운로드해야 합니다.")
                    continue
            
            # 필요한 컬럼만 선택하고 이름 변경
            if name == 'KOSPI':
                # 코스피: 종가, 시가, 고가, 저가, 거래량
                selected = filled_data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
                selected.columns = ['KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume']
            elif name == 'NASDAQ':
                # 나스닥: 종가만
                selected = filled_data[['Close']].copy()
                selected.columns = ['NASDAQ_Close']
            elif name == 'US10Y':
                # 미국 10년물 국채 금리: 종가만
                selected = filled_data[['Close']].copy()
                selected.columns = ['US10Y_Yield']
            elif name == 'VIX':
                # VIX: 종가만
                selected = filled_data[['Close']].copy()
                selected.columns = ['VIX_Close']
            elif name == 'USD_KRW':
                # 원/달러 환율: 종가만
                selected = filled_data[['Close']].copy()
                selected.columns = ['USD_KRW']
            
            # DataFrame 결합
            if combined_df is None:
                combined_df = selected
            else:
                combined_df = pd.concat([combined_df, selected], axis=1)
            
            print(f"✅ {name} 데이터 추가 완료")
        else:
            print(f"❌ {name} 데이터를 가져올 수 없습니다.")
    
    # 인덱스 이름 설정
    if combined_df is not None:
        combined_df.index.name = 'Date'
    
    return combined_df

# 예시 사용법
if __name__ == "__main__": 
    start_date = '2025-01-01'
    end_date = '2025-11-12'
    
    print("="*70)
    print("KOSPI 데이터셋 생성 시작")
    print("="*70)
    
    # 통합 데이터셋 생성
    dataset = create_combined_dataset(start_date, end_date)
    
    if dataset is not None and not dataset.empty:
        print("\n" + "="*70)
        print("데이터셋 생성 완료!")
        print("="*70)
        print(f"\n데이터 형태: {dataset.shape}")
        print(f"컬럼 목록: {list(dataset.columns)}")
        print(f"\n처음 5개 행:")
        print(dataset.head())
        print(f"\n마지막 5개 행:")
        print(dataset.tail())
        print(f"\n데이터 요약 통계:")
        print(dataset.describe())
        
        # NaN 체크
        if dataset.isna().any().any():
            print(f"\n⚠️ Warning: 최종 데이터에 {dataset.isna().sum().sum()}개의 NaN이 있습니다.")
            print(dataset.isna().sum())
        else:
            print("\n✅ 최종 데이터에 NaN이 없습니다!")
        
        # CSV 파일로 저장
        output_filename = f'kospi_dataset_{start_date}_to_{end_date}.csv'
        dataset.to_csv(output_filename, index=True, encoding='utf-8-sig', date_format='%Y-%m-%d')
        print(f"\n💾 데이터가 '{output_filename}' 파일로 저장되었습니다.")
    else:
        print("\n❌ 데이터셋 생성 실패")