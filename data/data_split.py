import pandas as pd
import os

def split_dataset(input_csv, output_dir='data'):
    """
    KOSPI 데이터셋을 학습/검증/평가 세트로 분할합니다.
    
    매개변수:
    input_csv (str): 입력 CSV 파일 경로
    output_dir (str): 출력 디렉토리 (기본값: 'data')
    
    반환값:
    dict: 각 데이터셋의 정보를 담은 딕셔너리
    """
    print("="*70)
    print("데이터셋 분할 시작")
    print("="*70)
    
    # CSV 파일 읽기
    print(f"\n📂 데이터 로딩 중: {input_csv}")
    df = pd.read_csv(input_csv, index_col='Date', parse_dates=True)
    print(f"✅ 전체 데이터 로딩 완료: {df.shape}")
    print(f"   날짜 범위: {df.index.min()} ~ {df.index.max()}")
    
    # 날짜 범위 정의
    train_start = '2010-01-01'
    train_end = '2021-12-31'
    
    valid_start = '2022-01-01'
    valid_end = '2023-12-31'
    
    test_start = '2024-01-01'
    test_end = '2025-10-31'
    
    # 데이터 분할
    print("\n" + "="*70)
    print("데이터 분할 중...")
    print("="*70)
    
    # Training Set (2010-2020)
    train_df = df.loc[train_start:train_end]
    print(f"\n✅ Training Set")
    print(f"   기간: {train_start} ~ {train_end}")
    print(f"   크기: {train_df.shape}")
    print(f"   실제 날짜 범위: {train_df.index.min()} ~ {train_df.index.max()}")
    
    # Validation Set (2021-2023)
    valid_df = df.loc[valid_start:valid_end]
    print(f"\n✅ Validation Set")
    print(f"   기간: {valid_start} ~ {valid_end}")
    print(f"   크기: {valid_df.shape}")
    print(f"   실제 날짜 범위: {valid_df.index.min()} ~ {valid_df.index.max()}")
    
    # Test Set (2024-2025)
    test_df = df.loc[test_start:test_end]
    print(f"\n✅ Test Set")
    print(f"   기간: {test_start} ~ {test_end}")
    print(f"   크기: {test_df.shape}")
    print(f"   실제 날짜 범위: {test_df.index.min()} ~ {test_df.index.max()}")
    
    # NaN 체크
    print("\n" + "="*70)
    print("데이터 품질 검사")
    print("="*70)
    
    for name, data in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"\n⚠️  {name} Set: {nan_count}개의 NaN 발견")
            print(data.isna().sum())
        else:
            print(f"\n✅ {name} Set: NaN 없음")
    
    # CSV 파일로 저장
    print("\n" + "="*70)
    print("CSV 파일 저장 중...")
    print("="*70)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.csv')
    valid_file = os.path.join(output_dir, 'valid.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_file, encoding='utf-8-sig', date_format='%Y-%m-%d')
    valid_df.to_csv(valid_file, encoding='utf-8-sig', date_format='%Y-%m-%d')
    test_df.to_csv(test_file, encoding='utf-8-sig', date_format='%Y-%m-%d')
    
    print(f"\n💾 Train Set 저장: {train_file}")
    print(f"💾 Valid Set 저장: {valid_file}")
    print(f"💾 Test Set 저장: {test_file}")
    
    # 결과 요약
    print("\n" + "="*70)
    print("분할 완료 요약")
    print("="*70)
    print(f"\n전체 데이터: {df.shape[0]}개 행")
    print(f"  ├─ Train:  {train_df.shape[0]}개 행 ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"  ├─ Valid:  {valid_df.shape[0]}개 행 ({valid_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"  └─ Test:   {test_df.shape[0]}개 행 ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    
    return {
        'train': train_df,
        'valid': valid_df,
        'test': test_df,
        'files': {
            'train': train_file,
            'valid': valid_file,
            'test': test_file
        }
    }


if __name__ == "__main__":
    # 입력 파일명 (data_gen.py에서 생성된 파일)
    input_csv = 'kospi_dataset_2010-01-01_to_2025-10-31.csv'
    
    # 파일 존재 확인
    if not os.path.exists(input_csv):
        print(f"❌ Error: '{input_csv}' 파일을 찾을 수 없습니다.")
        print("\n먼저 data_gen.py를 실행하여 데이터를 생성하세요:")
        print("  python data\\data_gen.py")
    else:
        # 데이터 분할 실행
        result = split_dataset(input_csv)
        
        print("\n" + "="*70)
        print("✅ 모든 작업이 완료되었습니다!")
        print("="*70)
