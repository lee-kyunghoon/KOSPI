import streamlit as st
import pandas as pd
import numpy as np
import torch
import yaml
import pickle
import os
import shutil
import zipfile
import tempfile
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from model.ae.model import KOSPIPredictor
from model.transformer.transformer import CNNTrans
import atexit


@st.cache_resource
def load_model(checkpoint_path, config):
    """모델 로드"""
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get('model_name', 'aecnn').lower()
    
    if model_name == 'aecnn':
        model = KOSPIPredictor(
            in_features=config['model']['in_features'],
            sae_hidden_dims=config['model']['sae_hidden_dims'],
            sae_latent_dim=config['model']['sae_latent_dim'],
            sae_noise_factor=config['model']['sae_noise_factor'],
            backbone=config['model']['backbone'],
            cnn_channels=config['model']['cnn_channels'],
            cnn_kernel_sizes=config['model']['cnn_kernel_size'],
            prediction_days=config['data']['prediction_days'],
            sequence_length=config['data']['sequence_length'],
            dropout=config['model']['dropout'],
            use_revin=config['model']['use_revin']
        )
    elif model_name == 'cnntrans':
        model = CNNTrans(
            input_features=config['model']['in_features'],
            output_seq_len=config['data']['prediction_days'],
            conv_out_channels=config['model'].get('conv_out_channels', 252),
            conv_kernel_size=config['model'].get('conv_kernel_size', 3),
            d_model=config['model'].get('d_model', 512),
            nhead=config['model'].get('nhead', 8),
            num_encoder_layers=config['model'].get('num_encoder_layers', 3),
            dim_feedforward=config['model'].get('dim_feedforward', 2048),
            dropout=config['model'].get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, model_name


@st.cache_resource
def load_scaler(checkpoint_dir, normalization_method):
    """스케일러 로드"""
    if normalization_method in ['robust', 'minmax']:
        scaler_path = os.path.join(checkpoint_dir, 'scaler_info.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
    return None


def get_last_trading_day(target_date):
    """주말/공휴일 처리: 가장 최근 거래일 반환"""
    while target_date.weekday() >= 5:
        target_date -= timedelta(days=1)
    return target_date


def get_next_trading_days(start_date, n_days=5):
    """다음 n개 거래일 반환 (주말 제외)"""
    trading_days = []
    current = start_date + timedelta(days=1)
    
    while len(trading_days) < n_days:
        if current.weekday() < 5:
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days


@st.cache_data(ttl=3600)
def fetch_market_data(end_date, sequence_length=30):
    """시장 데이터 다운로드 (prepare_dataset.py 로직 적용)"""
    end_date = get_last_trading_day(end_date)
    start_date = end_date - timedelta(days=sequence_length * 2)
    
    tickers = {
        'KOSPI': '^KS11',
        'NASDAQ': '^IXIC',
        'US10Y': '^TNX',
        'VIX': '^VIX',
        'USD_KRW': 'KRW=X'
    }
    
    combined_df = None
    kospi_dates = None
    
    for name, ticker in tickers.items():
        try:
            if name != 'KOSPI':
                adjusted_start = start_date - timedelta(days=1)
                data = yf.download(ticker, start=adjusted_start, end=end_date + timedelta(days=1), 
                                 progress=False, auto_adjust=True)
            else:
                data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), 
                                 progress=False, auto_adjust=True)
            
            if data.empty:
                st.error(f"{name} 데이터를 가져올 수 없습니다.")
                return None
            
            if name == 'KOSPI':
                kospi_dates = data.index
                filled_data = data
            else:
                if kospi_dates is not None:
                    data.index = data.index + pd.Timedelta(days=1)
                    data = data.reindex(kospi_dates)
                    data = data.ffill().bfill()
                    filled_data = data
                else:
                    st.error("KOSPI 데이터를 먼저 로드해야 합니다.")
                    return None
            
            if name == 'KOSPI':
                selected = filled_data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
                selected.columns = ['KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume']
            elif name == 'NASDAQ':
                selected = filled_data[['Close']].copy()
                selected.columns = ['NASDAQ_Close']
            elif name == 'US10Y':
                selected = filled_data[['Close']].copy()
                selected.columns = ['US10Y_Yield']
            elif name == 'VIX':
                selected = filled_data[['Close']].copy()
                selected.columns = ['VIX_Close']
            elif name == 'USD_KRW':
                selected = filled_data[['Close']].copy()
                selected.columns = ['USD_KRW']
            
            if combined_df is None:
                combined_df = selected
            else:
                combined_df = pd.concat([combined_df, selected], axis=1)
                
        except Exception as e:
            st.error(f"{name} 데이터 다운로드 실패: {e}")
            return None
    
    if combined_df.isna().any().any():
        combined_df = combined_df.ffill().bfill().fillna(0)
    
    return combined_df


def predict_next_5_days(model, data, scaler, config):
    """향후 5일 예측"""
    device = torch.device("cpu")
    sequence_length = config['data']['sequence_length']
    
    recent_data = data.iloc[-sequence_length:].values
    
    if scaler is not None:
        recent_data = scaler.transform(recent_data)
    
    input_tensor = torch.FloatTensor(recent_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if isinstance(model, CNNTrans):
            output = model(input_tensor)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output
        else:
            predictions = model(input_tensor, return_reconstruction=False)
    
    predictions = predictions.squeeze().cpu().numpy()
    
    return predictions


def cleanup_temp_checkpoints():
    temp_dir = 'temp_checkpoints'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

atexit.register(cleanup_temp_checkpoints)

# ==================== Streamlit UI ====================

st.set_page_config(
    page_title="KOSPI 5일 예측", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.title("KOSPI 5일 예측 시스템")

with st.sidebar:
    st.header("모델 설정")
    
    st.subheader("모델 선택")
    
    model_options = ["AECNN", "CNNTrans"]
    
    selected_model = st.radio(
        "모델 타입",
        model_options,
        index=0,
        horizontal=True,
        help="사용할 모델 타입을 선택하세요.",
        key="model_selector"
    )
    
    if 'last_selected_model' not in st.session_state:
        st.session_state['last_selected_model'] = selected_model
    
    if st.session_state['last_selected_model'] != selected_model:
        if 'predictions' in st.session_state:
            del st.session_state['predictions']
        if 'last_trading_day' in st.session_state:
            del st.session_state['last_trading_day']
        if 'next_trading_days' in st.session_state:
            del st.session_state['next_trading_days']
        if 'market_data' in st.session_state:
            del st.session_state['market_data']
        if 'model_name' in st.session_state:
            del st.session_state['model_name']
        if 'uploaded_checkpoint' in st.session_state:
            del st.session_state['uploaded_checkpoint']
        st.session_state['last_selected_model'] = selected_model
    
    st.session_state['selected_model_type'] = selected_model.lower()
    
    st.divider()
    
    upload_option = st.radio(
        "체크포인트 선택 방법",
        ["기존 폴더 선택", "ZIP 파일 업로드"]
    )
    
    if upload_option == "ZIP 파일 업로드":
        st.info("체크포인트 폴더를 압축(zip)하여 업로드하세요.")
        
        uploaded_file = st.file_uploader("ZIP 파일 업로드", type=['zip'])
        
        if uploaded_file is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'checkpoint.zip')
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                extracted_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
                
                if extracted_dirs:
                    source_dir = os.path.join(tmpdir, extracted_dirs[0])
                    model_folder = selected_model.lower()
                    target_dir = os.path.join('temp_checkpoints', model_folder)
                    
                    os.makedirs('temp_checkpoints', exist_ok=True)
                    
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    
                    shutil.copytree(source_dir, target_dir)
                    st.success(f"업로드 완료")
                    st.session_state['uploaded_checkpoint'] = target_dir
    
    model_folder = selected_model.lower()
    
    if upload_option == "ZIP 파일 업로드" and 'uploaded_checkpoint' in st.session_state:
        selected_checkpoint = st.session_state['uploaded_checkpoint']
        checkpoint_path = os.path.join(selected_checkpoint, 'best_model.pt')
        config_path = os.path.join(selected_checkpoint, 'config.yaml')
    else:
        checkpoint_base = 'checkpoints'
        
        if os.path.exists(os.path.join(checkpoint_base, model_folder)):
            checkpoint_path = os.path.join(checkpoint_base, model_folder, 'best_model.pt')
            config_path = os.path.join(checkpoint_base, model_folder, 'config.yaml')
            selected_checkpoint = os.path.join(checkpoint_base, model_folder)
        else:
            st.error(f"{model_folder} 폴더를 찾을 수 없습니다. ZIP 파일 업로드를 사용하세요.")
            st.stop()
    
    if not os.path.exists(checkpoint_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {checkpoint_path}")
        st.stop()
    
    if not os.path.exists(config_path):
        st.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        st.stop()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    st.divider()
    
    st.subheader("예측 날짜")
    target_date = st.date_input(
        "기준 날짜",
        value=datetime.now()
    )
    st.caption("※ 주말/공휴일인 경우 가장 최근 거래일 기준으로 예측합니다.")
    
    target_date = datetime.combine(target_date, datetime.min.time())
    
    if 'last_target_date' not in st.session_state:
        st.session_state['last_target_date'] = target_date
    
    if st.session_state['last_target_date'] != target_date:
        if 'predictions' in st.session_state:
            del st.session_state['predictions']
        if 'last_trading_day' in st.session_state:
            del st.session_state['last_trading_day']
        if 'next_trading_days' in st.session_state:
            del st.session_state['next_trading_days']
        if 'market_data' in st.session_state:
            del st.session_state['market_data']
        if 'status_messages' in st.session_state:
            del st.session_state['status_messages']
        st.session_state['last_target_date'] = target_date

st.divider()

last_trading_day = get_last_trading_day(target_date)
next_trading_days = get_next_trading_days(last_trading_day)

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("기준 거래일", last_trading_day.strftime('%Y-%m-%d (%a)'))

with col_info2:
    st.metric("예측 시작일", next_trading_days[0].strftime('%Y-%m-%d'))

with col_info3:
    st.metric("예측 종료일", next_trading_days[-1].strftime('%Y-%m-%d'))

if st.button("예측 실행", type="primary", use_container_width=True):
    with st.spinner("모델 로딩 중..."):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint['model_name'] = st.session_state.get('selected_model_type', 'aecnn')
            
            temp_checkpoint_path = checkpoint_path + '.temp'
            torch.save(checkpoint, temp_checkpoint_path)
            
            model, model_name = load_model(temp_checkpoint_path, config)
            scaler = load_scaler(selected_checkpoint, config['data']['normalization_method'])
            
            os.remove(temp_checkpoint_path)
            
            st.session_state['status_messages'] = [f"모델 로드 완료: {model_name.upper()}"]
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)
            st.stop()
    
    with st.spinner("시장 데이터 다운로드 중..."):
        try:
            market_data = fetch_market_data(last_trading_day, config['data']['sequence_length'])
        except Exception as e:
            st.error(f"데이터 다운로드 실패: {e}")
            st.stop()
    
    if market_data is None or len(market_data) < config['data']['sequence_length']:
        st.error("충분한 데이터를 가져올 수 없습니다.")
        st.stop()
    
    st.session_state['status_messages'].append(f"데이터 로드 완료: {len(market_data)}일")
    
    with st.spinner("예측 실행 중..."):
        try:
            predictions = predict_next_5_days(model, market_data, scaler, config)
            st.session_state['status_messages'].append("예측 완료!")
        except Exception as e:
            st.error(f"예측 실패: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    st.session_state['predictions'] = predictions
    st.session_state['last_trading_day'] = last_trading_day
    st.session_state['next_trading_days'] = next_trading_days
    st.session_state['market_data'] = market_data
    st.session_state['model_name'] = model_name

st.divider()

if 'predictions' in st.session_state:
    st.subheader("예측 결과")
    
    predictions = st.session_state['predictions']
    next_trading_days = st.session_state['next_trading_days']
    market_data = st.session_state['market_data']
    last_trading_day = st.session_state['last_trading_day']
    model_name = st.session_state['model_name']
    
    last_close = market_data.iloc[-1]['KOSPI_Close']
    
    pred_df = pd.DataFrame({
        '거래일': [d.strftime('%Y-%m-%d (%a)') for d in next_trading_days],
        '예측 종가': predictions,
        '전일 대비': [predictions[i] - (predictions[i-1] if i > 0 else last_close) 
                    for i in range(len(predictions))],
        '등락률(%)': [(predictions[i] - (predictions[i-1] if i > 0 else last_close)) / 
                     (predictions[i-1] if i > 0 else last_close) * 100 
                     for i in range(len(predictions))]
    })
    
    pred_df['예측 종가'] = pred_df['예측 종가'].round(2)
    pred_df['전일 대비'] = pred_df['전일 대비'].round(2)
    pred_df['등락률(%)'] = pred_df['등락률(%)'].round(2)
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        display_len = 7
        
        hist_data = market_data.iloc[-display_len:]['KOSPI_Close'].values
        hist_dates = market_data.iloc[-display_len:].index
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[d.strftime('%Y-%m-%d') for d in hist_dates],
            y=hist_data,
            mode='lines',
            name='실제 종가',
            line=dict(color='#4dabf7', width=2.5),
            hovertemplate='<b>%{x}</b><br>종가: %{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[last_trading_day.strftime('%Y-%m-%d')] + [d.strftime('%Y-%m-%d') for d in next_trading_days],
            y=[last_close] + list(predictions),
            mode='lines+markers',
            name='예측 종가',
            line=dict(color='#ff6b6b', width=2.5, dash='dash'),
            marker=dict(size=10, symbol='circle', line=dict(color='#ff6b6b', width=2)),
            hovertemplate='<b>%{x}</b><br>예측: %{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"KOSPI 종가 예측 ({model_name.upper()} 모델)",
            xaxis_title="날짜",
            yaxis_title="종가",
            hovermode='x unified',
            height=450,
            margin=dict(l=10, r=10, t=60, b=10),
            template="plotly"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_table:
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        avg_change = pred_df['등락률(%)'].mean()
        total_change = (predictions[-1] - last_close) / last_close * 100
        
        st.divider()
        
        if avg_change > 0:
            st.metric("일평균 등락률", f"+{avg_change:.2f}%", delta=None)
        else:
            st.metric("일평균 등락률", f"{avg_change:.2f}%", delta=None)
        
        if total_change > 0:
            st.metric("5일 누적 등락률", f"+{total_change:.2f}%", delta=None)
        else:
            st.metric("5일 누적 등락률", f"{total_change:.2f}%", delta=None)
    
    if 'status_messages' in st.session_state:
        st.divider()
        for msg in st.session_state['status_messages']:
            st.success(msg)
else:
    st.info("위의 '예측 실행' 버튼을 클릭하세요.")
    
    with st.expander("사용 방법"):
        st.write("1. 좌측 사이드바에서 모델 타입 선택 (AECNN 또는 CNNTrans)")
        st.write("2. 기준 날짜 설정 (주말/공휴일 자동 조정)")
        st.write("3. '예측 실행' 버튼 클릭")
        st.write("4. 향후 5거래일 예측 결과 확인")







