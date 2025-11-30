# KOSPI Stock Price Prediction Project
본 프로젝트는 KOSPI 주가 지수의 복잡성과 노이즈 문제를 해결하기 위해 딥러닝 모델을 활용한 연구입니다. 기존 LSTM 모델이 가진 긴 시퀀스 데이터의 중요도 판단 및 노이즈 제어 한계를 극복하고자, 두 가지 하이브리드 모델을 제안하여 **과거 N일 데이터를 기반으로 향후 5일간의 KOSPI 종가**를 예측합니다.

* **주요 접근법:**
    * **AE-CNN:** Stacked AutoEncoder를 통해 금융 데이터의 노이즈를 제거하고 1D CNN으로 추세를 학습.
    * **CNN-Transformer:** CNN 임베딩 후 Self-Attention을 통해 시계열 데이터 간의 중요도와 맥락을 파악.

## 1. Setup
프로젝트는 **Python** 환경에서 **PyTorch** 프레임워크를 사용하여 개발되었습니다.
(프로젝트 폴더 내에서)


```bash
conda create -n kkospi python=3.9 pip
conda activate kkospi
pip install -e .
```


## 2. Data preparation

### 2.1 데이터 생성 및 위치
프로젝트는 `yfinance`를 통해 수집된 데이터를 사용합니다. 아래 절차에 따라 학습 데이터를 준비해 주세요. (기본으로 제공됩니다.)

1.  **데이터 다운로드:**
    `data/prepare_dataset.py` 스크립트를 실행하여 원하는 기간을 설정하고 데이터를 수집합니다.
2.  **파일 위치:**
    수집된 데이터 파일을 각각 **`train.csv`**, **`test.csv`**, **`valid.csv`** 로 이름을 변경한 후, 프로젝트 루트의 **`data/`** 디렉토리 안에 위치시킵니다.

### 2.2 입력 특성
총 9개의 거시경제 및 시장 지표를 활용합니다.
* **KOSPI (5):** 종가, 시가, 고가, 저가, 거래량
* **Global (2):** 나스닥 지수(종가), 美 10년물 국채 금리
* **Sentiment/Macro (2):** VIX 지수, 원/달러 환율

## 3. Quick Start
데이터 준비가 완료되었다면, 아래 명령어를 통해 모델 학습을 시작할 수 있습니다.

```bash
python -m train
```

## 4. Launching Training

### 4.1 Configuration
학습에 필요한 하이퍼파라미터는 `config.yaml` 파일에서 관리됩니다.
* **config.yaml:** Epoch, Batch Size, Learning Rate, Model Architecture 등의 설정을 변경할 수 있습니다.
* **Random Search:** 최적의 파라미터 조합을 찾기 위해 아래 스크립트로 튜닝을 수행할 수 있습니다.
```bash
python random_search.py
```

### 4.2 Training
```bash
python -m train
```
튜닝을 통해 찾은 하이퍼파라미터는 **/test** 경로에 생기며, random_search_topk_results.json을 통해 상세 결과를 확인할 수 있습니다.
## 5. Launching Evaluations

학습이 완료된 후 저장된 모델 체크포인트(`.pt` 파일이 있는 폴더)를 사용하여 성능을 평가합니다.

### 5.1 Metric
모델의 수치적 정확도는 다음 지표로 측정됩니다.
* **MAPE** (Mean Absolute Percentage Error)
* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Squared Error)

### 5.2 테스트 실행
`--checkpoint` 인자에 테스트할 모델 폴더의 경로를 지정합니다.

```bash
python -m test --checkpoint path/to/pt/folder
```
### 5.3 예측
`data/` 경로에 `test_final.csv`로 이름으로 파일 추가 후, 실행
```bash
python -m predict --checkpoint path/to/pt/folder
```

## 6. Results
테스트 스크립트를 실행하면 예측 결과와 로그 파일이 생성됩니다.

* **저장 위치:** 지정한 체크포인트 경로 내의 `results/` 폴더에 저장됩니다.
    * Example: `path/to/pt/folder/results/`
* **내용:** 시계열 비교 그래프(Actual vs Predicted), Error Distribution 히스토그램, 평가지표 결과 텍스트 파일 등이 포함됩니다.

## 7. Contributors
* **이경훈:** 프로젝트 총괄, 제안 모델2(CNN-Transformer) 구현, 결과 분석
* **박종기:** 데이터 수집, 학습 전략 계획, GUI 구현, 보고서 작성
* **김지원:** 전처리, 제안 모델1(AE-CNN) 구현, 평가지표 계산