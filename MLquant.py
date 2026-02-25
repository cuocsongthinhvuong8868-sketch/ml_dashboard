import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime
import os
import time
import yfinance as yf

# ---------------------------------------------------------
# BẮT BUỘC: Khai báo API Key TRƯỚC khi import vnstock
# ---------------------------------------------------------
os.environ['VNSTOCK_API_KEY'] = "vnstock_17b56a86b930db526e25e8de447a0bfd"
from vnstock import Quote 

# Tích hợp Keras cho Mạng Nơ-ron Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & DANH MỤC
# ==============================================================================
st.set_page_config(page_title="Quant ML: Advanced Market Sentiment", layout="wide")

LOCAL_DATA_FILE = "market_data_lake.parquet"
ML_RESULTS_FILE = "ml_results_lake.parquet"

VN30_TICKERS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

OTHER_LARGE_CAPS = [
    'ACV', 'VGI', 'MCH', 'BSR', 'VEA', 'POW', 'BVH', 'PNJ', 'REE', 'EIB', 
    'MSB', 'OCB', 'KDH', 'NLG', 'KBC', 'IDC', 'VGC', 'GMD', 'FRT', 'CTR'
]

LARGE_CAPS = VN30_TICKERS + OTHER_LARGE_CAPS

MID_CAPS = [
    'VND', 'VCI', 'HCM', 'VIX', 'MBS', 'SHS', 'BSI', 'FTS', 'CTS', 
    'DIG', 'DXG', 'PDR', 'NVL', 'CEO', 'HDG', 'SZC', 'TCH',        
    'HSG', 'NKG', 'DGW', 'PET', 'VHC', 'ANV', 'IDI', 'DBC', 'PAN', 
    'HAH', 'VOS', 'PVT', 'PVD', 'PVS', 'CSV', 'DCM', 'DPM',        
    'PC1', 'LCG', 'HHV', 'VCG', 'FCN', 'CTD', 'VTP', 'GEG', 'GEX'  
]

PENNIES = [
    'HQC', 'SCR', 'ITA', 'DLG', 'HAG', 'HNG', 'TTF', 'QCG', 'JVC', 'AMV', 
    'TSC', 'FIT', 'HAR', 'LDG', 'OGC', 'VHG', 'PXT', 'PXI', 'KMR', 'VMD', 
    'SJF', 'KHG', 'CRE', 'TDC', 'IJC', 'HAX', 'ASM', 'BCG', 'HBC'
]

ALL_TICKERS = {ticker: 'Large' for ticker in LARGE_CAPS}
ALL_TICKERS.update({ticker: 'Mid' for ticker in MID_CAPS})
ALL_TICKERS.update({ticker: 'Penny' for ticker in PENNIES})

def export_csv_button(df, file_name, button_label="📥 Tải Báo cáo (CSV)"):
    csv = df.to_csv(index=False).encode('utf-8-sig') 
    st.download_button(label=button_label, data=csv, file_name=file_name, mime='text/csv')

# ==============================================================================
# 2. HỆ THỐNG QUẢN LÝ DỮ LIỆU (HYBRID ENGINE ĐƯỢC CẤY VÀO STREAMLIT)
# ==============================================================================
def sync_market_data(years=5, force_full=False):
    tickers_list = list(ALL_TICKERS.keys())
    end_date = datetime.date.today()
    req_start_date = end_date - datetime.timedelta(days=int(years*365.25))
    
    if os.path.exists(LOCAL_DATA_FILE) and not force_full:
        df_existing = pd.read_parquet(LOCAL_DATA_FILE)
    else:
        df_existing = pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()
    new_data = []
    
    # Chuẩn bị danh sách các mã cần tải
    tasks = []
    for ticker in tickers_list:
        if not df_existing.empty and ticker in df_existing['ticker'].values:
            last_date = pd.to_datetime(df_existing[df_existing['ticker'] == ticker]['time']).max().date()
            start_date = last_date + datetime.timedelta(days=1)
        else:
            start_date = req_start_date
            
        if start_date <= end_date:
            tasks.append((ticker, start_date))
            
    if not tasks:
        progress_bar.empty(); status_text.empty()
        return False, "Dữ liệu các mã đều đã đầy đủ, không cần tải thêm."
        
    end_str = end_date.strftime("%Y-%m-%d")
    failed_yahoo = []
    failed_all = []
    
    total_tasks = len(tasks)
    
    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: TẢI BẰNG YAHOO FINANCE (0.3s)
    # ---------------------------------------------------------
    for i, (ticker, start_date) in enumerate(tasks):
        start_str = start_date.strftime("%Y-%m-%d")
        status_text.markdown(f"**GĐ1 - Yahoo Finance:** Đang tải `{ticker}` ({i+1}/{total_tasks})...")
        
        success = False
        try:
            yf_ticker = ticker + ".VN"
            stock = yf.Ticker(yf_ticker)
            df_yf = stock.history(start=start_str, end=end_str)
            
            if not df_yf.empty:
                df_yf = df_yf.reset_index()
                date_col = 'Date' if 'Date' in df_yf.columns else df_yf.columns[0]
                df_yf['time'] = pd.to_datetime(df_yf[date_col]).dt.tz_localize(None).dt.normalize()
                df_yf['close'] = df_yf['Close']
                df_yf['volume'] = df_yf['Volume']
                df_yf['ticker'] = ticker
                new_data.append(df_yf[['time', 'close', 'volume', 'ticker']])
                success = True
        except:
            pass
            
        if success:
            time.sleep(0.3)
        else:
            failed_yahoo.append((ticker, start_str))
            
        # Update progress (Phase 1 accounts for 50% of the bar max if there are failures)
        progress_bar.progress((i + 1) / (total_tasks + len(failed_yahoo) if failed_yahoo else total_tasks))

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: TẢI BÙ BẰNG VNSTOCK KBS (1.1s)
    # ---------------------------------------------------------
    if failed_yahoo:
        total_fail = len(failed_yahoo)
        for i, (ticker, start_str) in enumerate(failed_yahoo):
            status_text.markdown(f"**GĐ2 - VNStock (KBS):** Đang vá dữ liệu `{ticker}` ({i+1}/{total_fail})...")
            success = False
            try:
                q = Quote(symbol=ticker, source='KBS')
                df = q.history(start=start_str, end=end_str)
                
                if df is not None and not df.empty and 'volume' in df.columns:
                    df['time'] = pd.to_datetime(df['time']).dt.normalize()
                    df['ticker'] = ticker
                    new_data.append(df[['time', 'close', 'volume', 'ticker']])
                    success = True
            except:
                pass
                
            time.sleep(1.1) # Bắt buộc nghỉ chống block
            
            if not success:
                failed_all.append(ticker)
                
            progress_bar.progress((total_tasks + i + 1) / (total_tasks + total_fail))

    progress_bar.empty(); status_text.empty()
    
    if failed_all:
        st.warning(f"⚠️ Các mã lỗi tuyệt đối (Đã thử mọi cách đều tạch): {', '.join(failed_all)}")
    elif failed_yahoo:
        st.success(f"🎉 VNStock đã vá thành công toàn bộ {len(failed_yahoo)} mã Yahoo bỏ sót!")
        
    if new_data:
        df_new = pd.concat(new_data)
        if not df_existing.empty and not force_full:
            df_final = pd.concat([df_existing, df_new]).drop_duplicates(subset=['time', 'ticker'])
        else: 
            df_final = df_new
        df_final.to_parquet(LOCAL_DATA_FILE)
        return True, f"Đã cập nhật dữ liệu thành công!"
        
    return False, "Có lỗi xảy ra, không lấy được dữ liệu mới."

def force_redownload_ticker(ticker_to_fix, years=15):
    """CÔNG CỤ PHẪU THUẬT: Xóa sạch mã cũ bị lỗi và tải lại bằng Hybrid Engine"""
    ticker_to_fix = ticker_to_fix.upper()
    
    if os.path.exists(LOCAL_DATA_FILE):
        df_data = pd.read_parquet(LOCAL_DATA_FILE)
        df_data = df_data[df_data['ticker'] != ticker_to_fix]
    else:
        df_data = pd.DataFrame()

    if os.path.exists(ML_RESULTS_FILE):
        df_ml = pd.read_parquet(ML_RESULTS_FILE)
        df_ml = df_ml[df_ml['Ticker'] != ticker_to_fix]
        df_ml.to_parquet(ML_RESULTS_FILE)

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(years*365.25))
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    success = False
    new_data = []
    
    # 1. Thử Yahoo trước
    try:
        yf_ticker = ticker_to_fix + ".VN"
        stock = yf.Ticker(yf_ticker)
        df_yf = stock.history(start=start_str, end=end_str)
        if not df_yf.empty:
            df_yf = df_yf.reset_index()
            date_col = 'Date' if 'Date' in df_yf.columns else df_yf.columns[0]
            df_yf['time'] = pd.to_datetime(df_yf[date_col]).dt.tz_localize(None).dt.normalize()
            df_yf['close'] = df_yf['Close']
            df_yf['volume'] = df_yf['Volume']
            df_yf['ticker'] = ticker_to_fix
            new_data.append(df_yf[['time', 'close', 'volume', 'ticker']])
            success = True
    except: pass
    
    if success:
        time.sleep(0.3)
    else:
        # 2. Vá bằng VNStock KBS
        try:
            q = Quote(symbol=ticker_to_fix, source='KBS')
            df = q.history(start=start_str, end=end_str)
            if df is not None and not df.empty and 'volume' in df.columns:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()
                df['ticker'] = ticker_to_fix
                new_data.append(df[['time', 'close', 'volume', 'ticker']])
                success = True
        except: pass
        time.sleep(1.1)
        
    if success and new_data:
        df_new = pd.concat(new_data)
        df_final = pd.concat([df_data, df_new]).drop_duplicates(subset=['time', 'ticker'])
        df_final.to_parquet(LOCAL_DATA_FILE)
        return True, f"Thành công! Đã thay máu sạch sẽ {years} năm dữ liệu cho {ticker_to_fix}."
    else:
        df_data.to_parquet(LOCAL_DATA_FILE) 
        return False, f"Thất bại! Cả VNStock và Yahoo đều không có dữ liệu cho {ticker_to_fix}."

@st.cache_data(show_spinner=False)
def load_and_preprocess_matrices():
    if not os.path.exists(LOCAL_DATA_FILE): return None
    df_long = pd.read_parquet(LOCAL_DATA_FILE)
    if df_long.empty: return None

    df_long = df_long.drop_duplicates(subset=['time', 'ticker'])
    close_matrix = df_long.pivot(index='time', columns='ticker', values='close').ffill()
    volume_matrix = df_long.pivot(index='time', columns='ticker', values='volume').fillna(0)
    
    returns_matrix = close_matrix.pct_change()
    
    rolling_std = returns_matrix.rolling(window=20).std()
    mean_vol = rolling_std.rolling(window=60).mean()
    std_vol = rolling_std.rolling(window=60).std()
    z_vol_matrix = (rolling_std - mean_vol) / std_vol.replace(0, np.nan)
    
    ma50_matrix = close_matrix.rolling(window=50).mean()
    breadth_series = (close_matrix > ma50_matrix).sum(axis=1) / close_matrix.shape[1]
    
    total_vol_series = volume_matrix.sum(axis=1)
    rank_vol_matrix = volume_matrix.rank(axis=1, ascending=False, method='first')
    top10_vol_series = volume_matrix[rank_vol_matrix <= 10].sum(axis=1)
    liquidity_drying_series = top10_vol_series / total_vol_series.replace(0, np.nan)
    
    dispersion_series = returns_matrix.std(axis=1)
    
    return {
        'close': close_matrix, 'returns': returns_matrix, 'z_vol': z_vol_matrix,
        'breadth': breadth_series, 'liquidity_drying': liquidity_drying_series,
        'return_dispersion': dispersion_series, 'all_dates': close_matrix.index
    }

def build_daily_features(matrices, date):
    if date not in matrices['all_dates']: return None
    df = pd.DataFrame({
        'Z_Volatility': matrices['z_vol'].loc[date],
        'Daily_Return': matrices['returns'].loc[date],
        'Close': matrices['close'].loc[date]
    })
    df['Market_Breadth'] = matrices['breadth'].loc[date]
    df['Liquidity_Drying'] = matrices['liquidity_drying'].loc[date]
    df['Return_Dispersion'] = matrices['return_dispersion'].loc[date]
    df['Cap'] = df.index.map(ALL_TICKERS)
    df['Ticker'] = df.index
    return df.dropna()

# ==============================================================================
# 3. LÕI AI & PRE-COMPUTATION (ML RESULTS LAKE)
# ==============================================================================
def run_ml_for_single_day(df_daily, historical_features_list, threshold_percentile=90):
    if len(df_daily) < 30: return None
    X_today = df_daily[['Z_Volatility', 'Market_Breadth', 'Liquidity_Drying', 'Return_Dispersion']]
    
    X_train_full = X_today if len(historical_features_list) < 5 else pd.concat(historical_features_list, ignore_index=True)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_today_scaled = scaler.transform(X_today)
    
    iso = IsolationForest(contamination=0.15, random_state=42)
    iso.fit(X_train_scaled) 
    df_daily['Iso_Anomaly'] = iso.predict(X_today_scaled)
    
    tf.get_logger().setLevel('ERROR')
    ae = Sequential([Input(shape=(4,)), Dense(8, activation='relu'), Dense(4, activation='linear')])
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_train_scaled, X_train_scaled, epochs=5, batch_size=64, verbose=0) 
    
    mse_history = np.mean(np.power(X_train_scaled - ae.predict(X_train_scaled, verbose=0), 2), axis=1)
    df_daily['Threshold'] = np.percentile(mse_history, threshold_percentile) 
    df_daily['MSE'] = np.mean(np.power(X_today_scaled - ae.predict(X_today_scaled, verbose=0), 2), axis=1)
    df_daily['Is_Flagged'] = (df_daily['Iso_Anomaly'] == -1) & (df_daily['MSE'] > df_daily['Threshold'])
    return df_daily

def sync_ml_results_lake(matrices):
    all_dates = matrices['all_dates']
    
    if os.path.exists(ML_RESULTS_FILE):
        df_ml = pd.read_parquet(ML_RESULTS_FILE)
        existing_dates = set(pd.to_datetime(df_ml['Date']).dt.date)
    else:
        df_ml = pd.DataFrame()
        existing_dates = set()

    dates_to_compute = [d for d in all_dates if d.date() not in existing_dates]

    if not dates_to_compute:
        return True, "Dữ liệu AI đã được tính toán đầy đủ cho mọi ngày!"

    progress_bar = st.progress(0)
    status_text = st.empty()
    new_results = []
    historical_features_window = []
    
    total_dates = len(dates_to_compute)
    
    for i, current_date in enumerate(all_dates):
        df_daily = build_daily_features(matrices, current_date)
        
        if df_daily is not None and len(df_daily) > 30:
            if current_date in dates_to_compute:
                status_text.markdown(f"**Đang tính AI (Batch):** {current_date.strftime('%Y-%m-%d')} ({len(new_results)+1}/{total_dates})")
                res_df = run_ml_for_single_day(df_daily.copy(), historical_features_window, threshold_percentile=90)
                if res_df is not None:
                    res_df['Date'] = current_date
                    new_results.append(res_df)
                    
            X_today = df_daily[['Z_Volatility', 'Market_Breadth', 'Liquidity_Drying', 'Return_Dispersion']].copy()
            historical_features_window.append(X_today)
            if len(historical_features_window) > 60:
                historical_features_window.pop(0)
                
            progress_bar.progress(min((i + 1) / len(all_dates), 1.0))

    progress_bar.empty(); status_text.empty()
    
    if new_results:
        df_new_ml = pd.concat(new_results)
        df_final_ml = pd.concat([df_ml, df_new_ml]).drop_duplicates(subset=['Date', 'Ticker']) if not df_ml.empty else df_new_ml
        df_final_ml.to_parquet(ML_RESULTS_FILE)
        return True, f"Đã tính toán bù AI cho {len(dates_to_compute)} ngày!"
    return False, "Không có ngày nào đủ dữ liệu để tính AI."

def run_micro_ai_for_ticker(ticker):
    if not os.path.exists(LOCAL_DATA_FILE): return pd.DataFrame()
    
    df_raw = pd.read_parquet(LOCAL_DATA_FILE)
    df_ticker = df_raw[df_raw['ticker'] == ticker].copy()
    if df_ticker.empty: return pd.DataFrame()
    
    df_ticker['time'] = pd.to_datetime(df_ticker['time'])
    df_ticker = df_ticker.sort_values('time')
    df_ticker.set_index('time', inplace=True)
    
    df_ticker['Return'] = df_ticker['close'].pct_change()
    df_ticker['Return_Z'] = (df_ticker['Return'] - df_ticker['Return'].rolling(60).mean()) / df_ticker['Return'].rolling(60).std()
    df_ticker['Volume_Z'] = (df_ticker['volume'] - df_ticker['volume'].rolling(60).mean()) / df_ticker['volume'].rolling(60).std()
    df_ticker['MA20_Dist'] = df_ticker['close'] / df_ticker['close'].rolling(20).mean() - 1
    
    df_ticker.dropna(inplace=True)
    
    if len(df_ticker) > 60:
        X = df_ticker[['Return_Z', 'Volume_Z', 'MA20_Dist']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso = IsolationForest(contamination=0.04, random_state=42)
        df_ticker['Micro_Anomaly'] = iso.fit_predict(X_scaled)
    else:
        df_ticker['Micro_Anomaly'] = 1
        
    df_ticker['Date'] = df_ticker.index
    df_ticker['Is_Flagged'] = df_ticker['Micro_Anomaly'] == -1
    return df_ticker

# ==============================================================================
# 4. GIAO DIỆN CHÍNH & SIDEBAR (ALL-IN-ONE)
# ==============================================================================
st.sidebar.title("🎛️ Data Manager")

st.sidebar.markdown("---")
st.sidebar.subheader("A. Xem Nhanh Hôm Nay")
if st.sidebar.button("⚡ Tải nhanh 3 tháng & Xử lý AI"):
    with st.spinner("Đang kéo 90 ngày dữ liệu và phân tích AI..."):
        succ1, msg1 = sync_market_data(years=0.25, force_full=True)
        st.cache_data.clear()
        if succ1:
            mat = load_and_preprocess_matrices()
            sync_ml_results_lake(mat)
            st.sidebar.success("Sẵn sàng! Hãy vào xem Menu A hoặc B.")
            time.sleep(1.5); st.rerun()
        else: st.sidebar.error(msg1)

st.sidebar.subheader("B. Khởi tạo Backtest (Lần đầu)")
fetch_years = st.sidebar.number_input("Nhập số năm:", min_value=1, max_value=20, value=15, step=1)
if st.sidebar.button(f"📥 1. Đồng bộ Dữ liệu ({fetch_years} năm)"):
    with st.spinner(f"Đang tải FULL {fetch_years} năm dữ liệu..."):
        succ, msg = sync_market_data(years=fetch_years, force_full=True)
        st.cache_data.clear()
        if succ: st.sidebar.success(msg); time.sleep(1); st.rerun()
        else: st.sidebar.error(msg)

if st.sidebar.button("🧠 2. Pre-computation (Xử lý AI)"):
    mat = load_and_preprocess_matrices()
    if mat is None: st.sidebar.error("Chưa có dữ liệu giá. Hãy bấm Bước 1.")
    else:
        with st.spinner("Đang chạy AI cho toàn bộ lịch sử (Sẽ mất vài phút)..."):
            succ, msg = sync_ml_results_lake(mat)
            if succ: st.sidebar.success(msg); time.sleep(1); st.rerun()
            else: st.sidebar.error(msg)

st.sidebar.markdown("---")
st.sidebar.subheader("C. Nhập Dữ Liệu & Cập Nhật")
st.sidebar.markdown("**1. Nạp file có sẵn (Upload):**")
uploaded_data = st.sidebar.file_uploader("Upload market_data", type=['parquet'])
uploaded_ml = st.sidebar.file_uploader("Upload ml_results", type=['parquet'])

if st.sidebar.button("📤 Xác nhận Nạp dữ liệu"):
    success_upload = False
    if uploaded_data:
        with open(LOCAL_DATA_FILE, "wb") as f: f.write(uploaded_data.getbuffer())
        success_upload = True
    if uploaded_ml:
        with open(ML_RESULTS_FILE, "wb") as f: f.write(uploaded_ml.getbuffer())
        success_upload = True
    if success_upload:
        st.cache_data.clear()
        st.sidebar.success("Đã nạp file thành công!")
        time.sleep(1.5); st.rerun()
    else: st.sidebar.warning("Vui lòng chọn ít nhất 1 file.")

st.sidebar.markdown("**2. Cập nhật phiên mới nhất:**")
if st.sidebar.button("🔄 Auto Update (Data + AI)"):
    with st.spinner("Đang tải dữ liệu phiên mới nhất và tính toán AI..."):
        succ1, msg1 = sync_market_data(years=1, force_full=False)
        st.cache_data.clear()
        mat = load_and_preprocess_matrices()
        if mat is not None:
            sync_ml_results_lake(mat)
            st.sidebar.success("Hệ thống đã cập nhật xong dữ liệu hôm nay!")
            time.sleep(1.5); st.rerun()
        else: st.sidebar.error("Lỗi đọc ma trận.")

st.sidebar.markdown("---")
st.sidebar.subheader("D. Sửa Lỗi Từng Mã")
fix_ticker = st.sidebar.text_input("Nhập mã bị kẹt (VD: ACB):").upper()
if st.sidebar.button("🛠️ Xóa & Tải lại mã này"):
    if fix_ticker:
        with st.spinner(f"Đang thay máu dữ liệu cho {fix_ticker}..."):
            succ, msg = force_redownload_ticker(fix_ticker, years=15)
            if succ:
                st.cache_data.clear()
                mat = load_and_preprocess_matrices()
                if mat is not None: sync_ml_results_lake(mat)
                st.sidebar.success(msg)
                time.sleep(1.5); st.rerun()
            else: st.sidebar.error(msg)
    else: st.sidebar.warning("Vui lòng nhập tên mã.")

st.sidebar.markdown("---")
st.sidebar.subheader("E. Lưu Trữ & Quản Trị")
if os.path.exists(LOCAL_DATA_FILE):
    with open(LOCAL_DATA_FILE, "rb") as file:
        st.sidebar.download_button(label="💾 Tải file Giá", data=file, file_name=LOCAL_DATA_FILE, mime="application/octet-stream")
if os.path.exists(ML_RESULTS_FILE):
    with open(ML_RESULTS_FILE, "rb") as file:
        st.sidebar.download_button(label="💾 Tải file AI", data=file, file_name=ML_RESULTS_FILE, mime="application/octet-stream")

if st.sidebar.button("🗑️ Xóa toàn bộ Cache & Lịch sử"):
    if os.path.exists(LOCAL_DATA_FILE): os.remove(LOCAL_DATA_FILE)
    if os.path.exists(ML_RESULTS_FILE): os.remove(ML_RESULTS_FILE)
    st.cache_data.clear()
    st.sidebar.success("Đã dọn sạch!")
    time.sleep(1); st.rerun()

st.sidebar.markdown("---")
menu = st.sidebar.radio("Chọn Module:", ["A. Micro-AI Radar (Mã lẻ)", "B. ML Market Scanner (Vĩ mô)", "C. Backtest Center"])

df_ml_lake = pd.read_parquet(ML_RESULTS_FILE) if os.path.exists(ML_RESULTS_FILE) else pd.DataFrame()

# ==========================================
# MODULE A: MICRO-AI RADAR (SOI MÃ LẺ)
# ==========================================
if menu == "A. Micro-AI Radar (Mã lẻ)":
    st.title("📡 Micro-AI Radar (Cảnh báo Dòng tiền Cá nhân hóa)")
    
    col_t, col_m, col_ma = st.columns([1, 1, 1])
    with col_t:
        ticker = st.text_input("Nhập mã CP:", value="MBB").upper()
    with col_m:
        mode = st.radio("Chế độ xem:", ["📅 Xem ngày cụ thể", "📉 Backtest"])
    with col_ma:
        ma_option = st.selectbox("Chọn đường MA hỗ trợ đọc vị:", [20, 60, 125, 252], index=0, format_func=lambda x: f"MA {x} phiên")
    
    df_ticker_micro = run_micro_ai_for_ticker(ticker)
    
    if df_ticker_micro.empty:
        st.error(f"Không tìm thấy dữ liệu cho mã {ticker}. Hãy dùng công cụ D 'Sửa Lỗi Từng Mã' ở thanh bên trái!")
    else:
        df_ticker_micro[f'MA_{ma_option}'] = df_ticker_micro['close'].rolling(window=ma_option).mean()
        
        if mode == "📅 Xem ngày cụ thể":
            scan_date = st.date_input("Ngày tra cứu:", value=df_ticker_micro['Date'].max().date())
            scan_dt = pd.Timestamp(scan_date)
            
            if scan_dt not in df_ticker_micro['Date']:
                st.warning("Ngày này không có giao dịch hoặc chưa được tải.")
            else:
                row = df_ticker_micro[df_ticker_micro['Date'] == scan_dt].iloc[0]
                st.subheader(f"Chẩn đoán Micro-AI: {ticker} (Ngày {scan_dt.date()})")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Giá đóng cửa", f"{row['close']:,.0f}")
                col2.metric("Return Z-Score", f"{row['Return_Z']:.2f}")
                col3.metric("Volume Z-Score", f"{row['Volume_Z']:.2f}")
                col4.metric(f"MA {ma_option} tham chiếu", f"{row[f'MA_{ma_option}']:,.0f}")
                
                if row['Is_Flagged']: 
                    st.error("🚨 CẢNH BÁO: Phát hiện dòng tiền gom/xả đột biến ở riêng cổ phiếu này!")
                else: 
                    st.success("✅ Trạng thái tự thân: Bình thường.")
                
                export_csv_button(df_ticker_micro, file_name=f"Report_Micro_AI_{ticker}.csv", button_label=f"📥 Tải Báo cáo Micro-AI {ticker} (CSV)")

        elif mode == "📉 Backtest":
            col_start, col_end = st.columns(2)
            start_date = pd.Timestamp(col_start.date_input("Từ ngày:", value=datetime.date(2022, 1, 1)))
            end_date = pd.Timestamp(col_end.date_input("Đến ngày:", value=datetime.date(2022, 12, 31)))
            
            df_plot = df_ticker_micro[(df_ticker_micro['Date'] >= start_date) & (df_ticker_micro['Date'] <= end_date)]
            
            if not df_plot.empty:
                export_csv_button(df_plot[df_plot['Is_Flagged'] == True], file_name=f"Report_Anomalies_{ticker}.csv", button_label=f"📥 Tải DS Báo động {ticker} (CSV)")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['close'], mode='lines', name='Giá', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot[f'MA_{ma_option}'], mode='lines', name=f'MA {ma_option}', line=dict(color='#ff7f0e', width=2)))
                
                flagged = df_plot[df_plot['Is_Flagged'] == True]
                fig.add_trace(go.Scatter(x=flagged['Date'], y=flagged['close'], mode='markers', name='Dòng tiền bất thường', marker=dict(color='red', size=8, symbol='x')))
                
                fig.update_layout(title=f"Lịch sử đột biến dòng tiền tự thân & MA {ma_option}: {ticker}", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không có dữ liệu cho khoảng thời gian này.")

# ==========================================
# MODULE B: ML MARKET SCANNER 
# ==========================================
elif menu == "B. ML Market Scanner (Vĩ mô)":
    st.title("🤖 ML Market Scanner (Nhận diện Đứt gãy Vĩ mô)")
    
    if df_ml_lake.empty:
         st.error("Kho AI trống! Hãy nạp file dữ liệu ở Data Manager.")
    else:
        scan_date = st.date_input("Ngày quét:", value=df_ml_lake['Date'].max().date())
        scan_dt = pd.Timestamp(scan_date)
        
        df_day = df_ml_lake[df_ml_lake['Date'] == scan_dt]
        if df_day.empty:
            st.warning("Ngày này chưa được tính AI. Hãy dùng nút 'Auto Update' ở Sidebar.")
        else:
            export_csv_button(df_day, file_name=f"Market_Scanner_{scan_dt.date()}.csv", button_label=f"📥 Tải toàn bộ Dữ liệu Vĩ mô ngày {scan_dt.date()} (CSV)")

            df_flagged = df_day[df_day['Is_Flagged'] == True]
            heat_rate = (len(df_flagged) / len(df_day)) * 100
            counts = df_flagged['Cap'].value_counts()
            total_flagged = len(df_flagged)
            large_cap_ratio = (counts.get('Large', 0) / total_flagged * 100) if total_flagged > 0 else 0
            
            st.subheader("📊 Chỉ số Vĩ mô & Dòng tiền")
            col1, col2, col3 = st.columns(3)
            col1.metric("Heat Rate", f"{heat_rate:.1f}%")
            col2.metric("Market Breadth", f"{df_day['Market_Breadth'].iloc[0]*100:.1f}%")
            col3.metric("Báo động Thanh khoản", f"{df_day['Liquidity_Drying'].iloc[0]*100:.1f}%")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Phân tán sinh lời", f"{df_day['Return_Dispersion'].iloc[0]:.4f}")
            col5.metric("Tổng mã bất thường", f"{total_flagged} mã")
            col6.metric("Tỉ lệ Large-cap / Flagged", f"{large_cap_ratio:.1f}%")
            
            if heat_rate >= 35.0:
                if large_cap_ratio >= 80: st.error("🚨 CÁ MẬP XẢ TRỤ. Rủi ro gãy cấu trúc cao.")
                elif large_cap_ratio <= 45: st.error("🚨 VỠ BONG BÓNG ĐẦU CƠ. Dòng tiền tháo chạy Midcap/Penny.")
                else: st.error("🚨 Áp lực bán tháo lan rộng (Đứt gãy hệ thống).")
            elif heat_rate >= 15.0: st.warning("⚠️ CẢNH BÁO SỚM: Bắt đầu có sự phân hóa mạnh.")
            else: st.success("✅ Trạng thái an toàn.")
            
            if not df_flagged.empty:
                show_df = df_flagged[['Ticker', 'Cap', 'Z_Volatility', 'MSE']].sort_values('MSE', ascending=False)
                show_df['Z_Volatility'] = show_df['Z_Volatility'].apply(lambda x: f"{x:.2f}")
                show_df['MSE'] = show_df['MSE'].apply(lambda x: f"{x:.4f}")
                st.table(show_df.set_index('Ticker'))

# ==========================================
# MODULE C: BACKTEST CENTER 
# ==========================================
elif menu == "C. Backtest Center":
    st.title("🤖 Backtest Center (Chu kỳ Vĩ mô)")
    
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Từ ngày:", value=datetime.date(2022, 1, 1))
    end_date = col_end.date_input("Đến ngày:", value=datetime.date(2022, 12, 31))
    ema_span = st.sidebar.slider("Làm mượt EMA (Số phiên):", min_value=1, max_value=20, value=5)
    
    if df_ml_lake.empty:
        st.error("Kho AI trống! Hãy nạp file dữ liệu ở Data Manager.")
    else:
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df_filtered = df_ml_lake[(df_ml_lake['Date'] >= start_dt) & (df_ml_lake['Date'] <= end_dt)]
        
        if df_filtered.empty:
            st.warning("Không có dữ liệu. Vui lòng kiểm tra lại quá trình Đồng bộ.")
        else:
            grouped = df_filtered.groupby('Date')
            results_history = []
            
            for date, group in grouped:
                heat_rate = (group['Is_Flagged'].sum() / len(group)) * 100
                large_cap_flagged = group[group['Is_Flagged'] & (group['Cap'] == 'Large')].shape[0]
                total_flagged = group['Is_Flagged'].sum()
                
                results_history.append({
                    'Date': date, 'Heat_Rate': heat_rate,
                    'Market_Breadth': group['Market_Breadth'].iloc[0] * 100,
                    'Large_Cap_Flagged': large_cap_flagged, 'Total_Flagged': total_flagged
                })
                
            df_res = pd.DataFrame(results_history)
            df_res[f'Heat_Rate_EMA_{ema_span}'] = df_res['Heat_Rate'].ewm(span=ema_span, adjust=False).mean()
            df_res['Large_Cap_Ratio'] = (df_res['Large_Cap_Flagged'] / df_res['Total_Flagged'].replace(0, 1)) * 100
            df_res.loc[df_res['Total_Flagged'] == 0, 'Large_Cap_Ratio'] = 0 
            df_res[f'Ratio_EMA_{ema_span}'] = df_res['Large_Cap_Ratio'].ewm(span=ema_span, adjust=False).mean()
            
            export_csv_button(df_res, file_name=f"Backtest_Macro_{start_dt.date()}_to_{end_dt.date()}.csv", button_label="📥 Tải Báo cáo Chu kỳ Vĩ mô (CSV)")
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_res['Date'], y=df_res['Heat_Rate'], mode='lines', name='Raw Heat Rate', line=dict(color='black', dash='dash', width=1)))
            fig1.add_trace(go.Scatter(x=df_res['Date'], y=df_res[f'Heat_Rate_EMA_{ema_span}'], mode='lines', name=f'Heat Rate (EMA {ema_span})', line=dict(color='#ff4b4b', width=3)))
            fig1.add_hline(y=35.0, line_width=2, line_dash="dash", line_color="red", annotation_text="> 35.0", annotation_position="top left", annotation_font=dict(color="red", size=12))
            max_heat = max(df_res['Heat_Rate'].max(), 50)
            fig1.add_hrect(y0=35.0, y1=max_heat, line_width=0, fillcolor="red", opacity=0.1)
            fig1.add_hline(y=15.0, line_width=1.5, line_dash="dot", line_color="orange", annotation_text="> 15.0", annotation_position="bottom left", annotation_font=dict(color="orange", size=12))
            fig1.add_hline(y=10.0, line_width=2, line_dash="dash", line_color="green", annotation_text="< 10.0", annotation_position="bottom left", annotation_font=dict(color="green", size=12))
            fig1.add_hrect(y0=0, y1=10.0, line_width=0, fillcolor="green", opacity=0.1)
            fig1.add_trace(go.Scatter(x=df_res['Date'], y=df_res['Market_Breadth'], mode='lines', name='Market Breadth (% > MA50)', yaxis='y2', line=dict(color='#1f77b4', width=2)))
            fig1.update_layout(title="1. Rủi ro Đảo chiều Vĩ mô", yaxis=dict(title="Heat Rate (%)"), yaxis2=dict(title="Market Breadth (%)", overlaying="y", side="right"), hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_res['Date'], y=df_res['Large_Cap_Ratio'], name='Tỉ lệ ngày lẻ', marker_color='rgba(156, 114, 186, 0.4)'))
            fig2.add_trace(go.Scatter(x=df_res['Date'], y=df_res[f'Ratio_EMA_{ema_span}'], mode='lines', name=f'Xu hướng cấu trúc (EMA {ema_span})', line=dict(color='#6a0dad', width=3)))
            fig2.add_hline(y=80, line_width=2, line_dash="solid", line_color="red", annotation_text="80%", annotation_font=dict(color="red"))
            fig2.add_hline(y=61.5, line_width=1.5, line_dash="dash", line_color="black", annotation_text="61.5%", annotation_font=dict(color="black"))
            fig2.add_hline(y=45, line_width=2, line_dash="solid", line_color="blue", annotation_text="45%", annotation_position="bottom right", annotation_font=dict(color="blue"))
            fig2.update_layout(title="2. Định vị Nguồn gốc Rủi ro (Large-cap vs Mid/Penny)", yaxis=dict(title="Tỉ lệ Large-cap / Flagged (%)", range=[0, 105]), hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
