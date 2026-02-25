import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from vnstock import Quote

# 1. PAGE CONFIG PHẢI NẰM NGAY ĐÂY (TRƯỚC MỌI THỨ CỦA STREAMLIT)
st.set_page_config(page_title="Quant ML: Advanced Market Sentiment", layout="wide")

# 2. HÀM KIỂM TRA MẬT KHẨU CÓ NÚT ĐĂNG NHẬP
def check_password():
    """Hàm kiểm tra mật khẩu bằng Form và nút Submit."""
    
    # Nếu đã đăng nhập đúng từ trước, cho qua luôn
    if st.session_state.get("password_correct", False):
        return True

    # Tạo một hộp thoại (form) đăng nhập
    with st.form("login_form"):
        st.subheader("🔒 Đăng nhập hệ thống Quant ML")
        password_input = st.text_input("Vui lòng nhập mật khẩu (Token) để truy cập:", type="password")
        
        # Nút bấm Đăng nhập
        submit_button = st.form_submit_button("Đăng nhập")

    # Xử lý sự kiện khi người dùng bấm nút
    if submit_button:
        # Kiểm tra mật khẩu (lấy từ secrets.toml)
        if password_input == st.secrets["passwords"]["app_password"]:
            st.session_state["password_correct"] = True
            st.rerun() # Tải lại trang ngay lập tức để ẩn form và hiện app
        else:
            st.error("😕 Mật khẩu không chính xác. Vui lòng thử lại!")
            
    return False

# 3. LẬP CHỐT CHẶN (BARIE) BẰNG ST.STOP()
if not check_password():
    st.stop() # Chặn toàn bộ code bên dưới nếu chưa đăng nhập đúng

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & DANH MỤC (Chuẩn hóa vốn hóa 2026)
# ==============================================================================

os.environ['VNSTOCK_API_KEY'] = "vnstock_17b56a86b930db526e25e8de447a0bfd"
LOCAL_DATA_FILE = "market_data_lake.parquet"

# 1.1 Rổ VN30 (Cập nhật mới nhất)
VN30_TICKERS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# 1.2 Top Vốn hóa lớn / Thanh khoản cao ngoài VN30 (HOSE, HNX, UPCOM)
OTHER_LARGE_CAPS = [
    'ACV', 'VGI', 'MCH', 'BSR', 'VEA', 'POW', 'BVH', 'PNJ', 'REE', 'EIB', 
    'MSB', 'OCB', 'KDH', 'NLG', 'KBC', 'IDC', 'VGC', 'GMD', 'FRT', 'CTR'
]

# Gộp chung thành rổ Large-cap cho mô hình
LARGE_CAPS = VN30_TICKERS + OTHER_LARGE_CAPS

# RỔ MID_CAPS: Vốn hóa vừa và mức độ biến động (Beta) cao
MID_CAPS = [
    'VND', 'VCI', 'HCM', 'VIX', 'MBS', 'SHS', 'BSI', 'FTS', 'CTS', # Chứng khoán
    'DIG', 'DXG', 'PDR', 'NVL', 'CEO', 'HDG', 'SZC', 'TCH',        # Bất động sản
    'HSG', 'NKG', 'DGW', 'PET', 'VHC', 'ANV', 'IDI', 'DBC', 'PAN', # Bán lẻ, Thủy sản, Nông nghiệp
    'HAH', 'VOS', 'PVT', 'PVD', 'PVS', 'CSV', 'DCM', 'DPM',        # Cảng, Dầu khí, Phân bón
    'PC1', 'LCG', 'HHV', 'VCG', 'FCN', 'CTD', 'VTP', 'GEG', 'GEX'  # Xây dựng, Đầu tư công, Điện
]

# RỔ PENNIES: Vốn hóa nhỏ, tính đầu cơ cao
PENNIES = [
    'HQC', 'SCR', 'ITA', 'DLG', 'HAG', 'HNG', 'TTF', 'QCG', 'JVC', 'AMV', 
    'TSC', 'FIT', 'HAR', 'LDG', 'OGC', 'VHG', 'PXT', 'PXI', 'KMR', 'VMD', 
    'SJF', 'KHG', 'CRE', 'TDC', 'IJC', 'HAX', 'ASM', 'BCG', 'HBC'
]

ALL_TICKERS = {ticker: 'Large' for ticker in LARGE_CAPS}
ALL_TICKERS.update({ticker: 'Mid' for ticker in MID_CAPS})
ALL_TICKERS.update({ticker: 'Penny' for ticker in PENNIES})

# ==============================================================================
# 2. HỆ THỐNG QUẢN LÝ DỮ LIỆU (LOCAL DATA LAKE) & MA TRẬN
# ==============================================================================

def sync_market_data(years=5):
    """Tải dữ liệu mới nhất từ VNStock và lưu vào file Local Parquet"""
    tickers_list = list(ALL_TICKERS.keys())
    end_date = datetime.date.today()
    
    # Kiểm tra xem đã có data cũ chưa để tải bù (Incremental Update)
    if os.path.exists(LOCAL_DATA_FILE):
        try:
            df_existing = pd.read_parquet(LOCAL_DATA_FILE)
            last_date = pd.to_datetime(df_existing['time']).max().date()
            start_date = last_date + datetime.timedelta(days=1)
        except:
            df_existing = pd.DataFrame()
            start_date = end_date - datetime.timedelta(days=int(years*365.25))
    else:
        df_existing = pd.DataFrame()
        start_date = end_date - datetime.timedelta(days=int(years*365.25))

    if start_date > end_date:
        return True, "Dữ liệu Local đã ở trạng thái mới nhất!"

    progress_bar = st.progress(0)
    status_text = st.empty()
    new_data = []
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    for i, ticker in enumerate(tickers_list):
        status_text.markdown(f"**Đang tải/Cập nhật dữ liệu:** {ticker} ({i+1}/{len(tickers_list)})")
        try:
            q = Quote(symbol=ticker, source='KBS')
            df = q.history(start=start_str, end=end_str)
            if df is not None and not df.empty and 'volume' in df.columns:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()
                df['ticker'] = ticker
                new_data.append(df[['time', 'close', 'volume', 'ticker']])
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers_list))
        time.sleep(1.2) # Tránh bị rate limit
        
    progress_bar.empty()
    status_text.empty()
    
    if new_data:
        df_new = pd.concat(new_data)
        if not df_existing.empty:
            df_final = pd.concat([df_existing, df_new]).drop_duplicates(subset=['time', 'ticker'])
        else:
            df_final = df_new
        # Lưu vào Local Storage
        df_final.to_parquet(LOCAL_DATA_FILE)
        return True, f"Cập nhật thành công từ {start_str} đến {end_str}!"
    else:
        return False, "Không có dữ liệu mới nào được tải về."

@st.cache_data(show_spinner=False)
def load_and_preprocess_matrices(data_mode):
    """Tải dữ liệu và xử lý thành Ma trận (OHLCV Matrix) bằng Vectơ hóa"""
    if data_mode == "💾 Local (Siêu tốc - Khuyên dùng)" and os.path.exists(LOCAL_DATA_FILE):
        df_long = pd.read_parquet(LOCAL_DATA_FILE)
    else:
        return None # Yêu cầu người dùng đồng bộ dữ liệu
        
    if df_long.empty: return None

    # 1. Chuyển sang định dạng Ma Trận (Cột là Ticker, Dòng là Date)
    df_long = df_long.drop_duplicates(subset=['time', 'ticker'])
    close_matrix = df_long.pivot(index='time', columns='ticker', values='close').ffill()
    volume_matrix = df_long.pivot(index='time', columns='ticker', values='volume').fillna(0)
    
    # 2. TÍNH TOÁN VECTƠ HÓA TOÀN THỊ TRƯỜNG (SIÊU TỐC)
    returns_matrix = close_matrix.pct_change()
    
    # Z-Volatility
    rolling_std = returns_matrix.rolling(window=20).std()
    mean_vol = rolling_std.rolling(window=60).mean()
    std_vol = rolling_std.rolling(window=60).std()
    z_vol_matrix = (rolling_std - mean_vol) / std_vol.replace(0, np.nan)
    
    # Market Breadth
    ma50_matrix = close_matrix.rolling(window=50).mean()
    breadth_series = (close_matrix > ma50_matrix).sum(axis=1) / close_matrix.shape[1]
    
    # Liquidity Drying (Top 10 Volume / Total Volume)
    total_vol_series = volume_matrix.sum(axis=1)
    rank_vol_matrix = volume_matrix.rank(axis=1, ascending=False, method='first')
    top10_vol_series = volume_matrix[rank_vol_matrix <= 10].sum(axis=1)
    liquidity_drying_series = top10_vol_series / total_vol_series.replace(0, np.nan)
    
    # Return Dispersion
    dispersion_series = returns_matrix.std(axis=1)
    
    return {
        'close': close_matrix,
        'returns': returns_matrix,
        'z_vol': z_vol_matrix,
        'breadth': breadth_series,
        'liquidity_drying': liquidity_drying_series,
        'return_dispersion': dispersion_series,
        'all_dates': close_matrix.index
    }

def build_daily_features(matrices, date):
    """Trích xuất 1 lát cắt (Cross-section) của ma trận cho 1 ngày cụ thể"""
    if date not in matrices['all_dates']: return None
    
    df = pd.DataFrame({
        'Z_Volatility': matrices['z_vol'].loc[date],
        'Daily_Return': matrices['returns'].loc[date],
        'Close': matrices['close'].loc[date]
    })
    
    # Gắn các chỉ số vĩ mô của ngày hôm đó vào từng mã
    df['Market_Breadth'] = matrices['breadth'].loc[date]
    df['Liquidity_Drying'] = matrices['liquidity_drying'].loc[date]
    df['Return_Dispersion'] = matrices['return_dispersion'].loc[date]
    
    df['Cap'] = df.index.map(ALL_TICKERS)
    df['Ticker'] = df.index
    
    # Loại bỏ các mã chưa niêm yết hoặc thiếu dữ liệu trong ngày này
    return df.dropna()

# ==============================================================================
# 3. LÕI AI ML ENGINE
# ==============================================================================
def run_ml_detection(df_daily, historical_features_list, threshold_percentile=90):
    if len(df_daily) < 30: 
        return pd.DataFrame(), 0
    
    X_today = df_daily[['Z_Volatility', 'Market_Breadth', 'Liquidity_Drying', 'Return_Dispersion']]
    
    if len(historical_features_list) < 5: 
        X_train_full = X_today
    else:
        X_train_full = pd.concat(historical_features_list, ignore_index=True)
        
    scaler = StandardScaler()
    scaler.fit(X_train_full)
    
    X_train_scaled = scaler.transform(X_train_full)
    X_today_scaled = scaler.transform(X_today)
    
    iso = IsolationForest(contamination=0.15, random_state=42)
    iso.fit(X_train_scaled)
    df_daily['Iso_Anomaly'] = iso.predict(X_today_scaled)
    
    ae = Sequential([
        Input(shape=(4,)), 
        Dense(8, activation='relu'), 
        Dense(4, activation='linear')
    ])
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_train_scaled, X_train_scaled, epochs=5, batch_size=64, verbose=0) 
    
    mse_history = np.mean(np.power(X_train_scaled - ae.predict(X_train_scaled, verbose=0), 2), axis=1)
    mse_threshold = np.percentile(mse_history, threshold_percentile) 
    
    df_daily['MSE'] = np.mean(np.power(X_today_scaled - ae.predict(X_today_scaled, verbose=0), 2), axis=1)
    
    df_flagged = df_daily[(df_daily['Iso_Anomaly'] == -1) & (df_daily['MSE'] > mse_threshold)].copy()
    
    heat_rate = (len(df_flagged) / len(df_daily) * 100) if len(df_daily) > 0 else 0
    return df_flagged, heat_rate

# ==============================================================================
# 4. GIAO DIỆN CHÍNH & SIDEBAR
# ==============================================================================
st.sidebar.title("🎛️ Quant Control Center")

# Quản lý Data Lake
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Quản lý Dữ liệu")

# Thêm Tùy chỉnh số năm
fetch_years = st.sidebar.number_input("Số năm dữ liệu muốn lấy:", min_value=1, max_value=20, value=5, step=1)

data_mode = st.sidebar.radio("Nguồn Dữ liệu:", ["💾 Local (Siêu tốc - Khuyên dùng)", "🌐 Online (Cần đồng bộ)"])

if st.sidebar.button(f"🔄 Đồng bộ VNStock ({fetch_years} năm)"):
    with st.spinner(f"Đang đồng bộ {fetch_years} năm dữ liệu với VNStock..."):
        success, msg = sync_market_data(years=fetch_years)
        if success:
            st.sidebar.success(msg)
            time.sleep(1)
            st.rerun() # Refresh lại để cập nhật ma trận
        else:
            st.sidebar.error(msg)

# NÚT XÓA CACHE VÀ RESET
if st.sidebar.button("🗑️ Xóa dữ liệu Local (Reset)"):
    if os.path.exists(LOCAL_DATA_FILE):
        os.remove(LOCAL_DATA_FILE)
        st.cache_data.clear() # Dọn dẹp cache của Streamlit
        st.sidebar.success("Đã xóa file dữ liệu cũ và làm sạch Cache!")
        time.sleep(1)
        st.rerun()
    else:
        st.sidebar.warning("Không có file dữ liệu nào để xóa.")

if data_mode == "💾 Local (Siêu tốc - Khuyên dùng)":
    if os.path.exists(LOCAL_DATA_FILE):
        file_size = os.path.getsize(LOCAL_DATA_FILE) / (1024 * 1024)
        st.sidebar.caption(f"Trạng thái: ✅ Đã có Local Data ({file_size:.1f} MB)")
    else:
        st.sidebar.caption("Trạng thái: ❌ Chưa có Local Data. Hãy bấm Đồng bộ!")
        st.sidebar.info("Lưu ý: Nếu bạn vừa cập nhật danh sách mã cổ phiếu, hãy bấm 'Xóa dữ liệu Local' trước khi Đồng bộ lại.")

st.sidebar.markdown("---")
menu = st.sidebar.radio("Chọn Module:", ["A. Sentiment Radar", "B. ML Market Scanner", "C. Backtest Center"])

# KHỞI TẠO MA TRẬN DỮ LIỆU
matrices = None
if menu in ["B. ML Market Scanner", "C. Backtest Center"]:
    matrices = load_and_preprocess_matrices(data_mode)


if menu == "A. Sentiment Radar":
    st.title("📡 Sentiment Radar")
    st.info("Tính năng xem mã lẻ đang bảo trì để tích hợp với Data Lake mới.")

elif menu == "B. ML Market Scanner":
    st.title("🤖 ML Market Scanner (Nhận diện Stress Ngầm)")
    scan_date = st.sidebar.date_input("Ngày quét:", value=datetime.date.today())
    
    if st.button("🚀 Kích hoạt AI dual-verification"):
        if matrices is None:
            st.error("Chưa có dữ liệu Ma trận! Hãy vào thanh bên (Sidebar) để đồng bộ dữ liệu VNStock.")
        else:
            scan_dt = pd.Timestamp(scan_date)
            all_dates = matrices['all_dates']
            
            if scan_dt not in all_dates:
                # Tìm ngày giao dịch gần nhất trước ngày quét
                valid_dates = all_dates[all_dates <= scan_dt]
                if valid_dates.empty:
                    st.error("Dữ liệu không đủ hoặc ngày quét quá xa trong quá khứ.")
                    st.stop()
                scan_dt = valid_dates[-1]
                st.warning(f"Ngày {scan_date} không có giao dịch. Tự động lùi về ngày giao dịch gần nhất: {scan_dt.date()}")
            
            with st.spinner(f"Khởi động AI Engine & Trích xuất 60 phiên gần nhất..."):
                target_idx = all_dates.get_loc(scan_dt)
                start_idx = max(0, target_idx - 60)
                window_dates = all_dates[start_idx : target_idx + 1]
                
                historical_features_window = []
                df_daily_target = pd.DataFrame()
                
                for d in window_dates:
                    df_d = build_daily_features(matrices, d)
                    if df_d is not None and len(df_d) > 30:
                        if d < scan_dt:
                            historical_features_window.append(df_d[['Z_Volatility', 'Market_Breadth', 'Liquidity_Drying', 'Return_Dispersion']].copy())
                        elif d == scan_dt:
                            df_daily_target = df_d
                
                if not df_daily_target.empty:
                    df_flagged, heat_rate = run_ml_detection(df_daily_target, historical_features_window, threshold_percentile=90)
                    
                    counts = df_flagged['Cap'].value_counts() if not df_flagged.empty else {}
                    total_flagged = len(df_flagged)
                    large_cap_flagged = counts.get('Large', 0)
                    large_cap_ratio = (large_cap_flagged / total_flagged * 100) if total_flagged > 0 else 0
                    
                    st.markdown("---")
                    st.subheader("📊 Chỉ số Vĩ mô & Sức khỏe Dòng tiền")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Độ nóng (Heat Rate)", f"{heat_rate:.1f}%")
                    col2.metric("Độ rộng (Market Breadth)", f"{df_daily_target['Market_Breadth'].iloc[0]*100:.1f}%")
                    col3.metric("Báo động Thanh khoản", f"{df_daily_target['Liquidity_Drying'].iloc[0]*100:.1f}%")
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Độ phân tán sinh lời", f"{df_daily_target['Return_Dispersion'].iloc[0]:.4f}")
                    col5.metric("Tổng mã bất thường", f"{total_flagged} mã")
                    col6.metric("Tỉ lệ Large-cap / Flagged", f"{large_cap_ratio:.1f}%")
                    
                    # LOGIC CẢNH BÁO
                    if heat_rate >= 35.0:
                        if large_cap_ratio >= 80:
                            st.error("🚨 CHẨN ĐOÁN: CÁ MẬP XẢ TRỤ. Dòng tiền lớn đang rút khỏi nhóm vốn hóa lớn. Rủi ro gãy cấu trúc cao.")
                        elif large_cap_ratio <= 45:
                            st.error("🚨 CHẨN ĐOÁN: VỠ BONG BÓNG ĐẦU CƠ. Dòng tiền tháo chạy hoảng loạn khỏi nhóm Midcap/Penny.")
                        else:
                            st.error("🚨 CHẨN ĐOÁN: Áp lực bán tháo lan rộng đồng đều trên toàn thị trường (Đứt gãy hệ thống).")
                    elif heat_rate >= 15.0:
                        st.warning("⚠️ CẢNH BÁO SỚM: Bắt đầu có sự phân hóa mạnh hoặc rủi ro phân phối ngầm (Warning Zone).")
                    else:
                        st.success("✅ Trạng thái an toàn (Safe Zone): Không phát hiện sự đứt gãy cấu trúc lớn.")
                    
                    if not df_flagged.empty:
                        st.markdown("---")
                        st.subheader("🚩 DANH SÁCH MÃ BỊ AI TUÝT CÒI (ANOMALY)")
                        show_df = df_flagged[['Ticker', 'Cap', 'Z_Volatility', 'MSE']].sort_values('MSE', ascending=False)
                        show_df['Z_Volatility'] = show_df['Z_Volatility'].apply(lambda x: f"{x:.2f}")
                        show_df['MSE'] = show_df['MSE'].apply(lambda x: f"{x:.2f}")
                        st.table(show_df.set_index('Ticker'))
                else:
                    st.error("Không đủ dữ liệu giao dịch cho ngày này.")

elif menu == "C. Backtest Center":
    st.title("🤖 Backtest Center (Phân tích Chu kỳ & Dòng tiền)")
    
    col_start, col_end = st.sidebar.columns(2)
    start_date = col_start.date_input("Từ ngày:", value=datetime.date(2022, 1, 1))
    end_date = col_end.date_input("Đến ngày:", value=datetime.date(2022, 12, 31))
    ema_span = st.sidebar.slider("Làm mượt EMA (Số phiên):", min_value=1, max_value=20, value=5)
    
    if st.button("🚀 Chạy Backtest (Ma trận Siêu tốc)"):
        if matrices is None:
            st.error("Chưa có dữ liệu Ma trận! Hãy vào thanh bên (Sidebar) để đồng bộ dữ liệu VNStock.")
        else:
            with st.spinner("Đang chạy mô hình AI/ML qua Ma trận dữ liệu..."):
                results_history = []
                all_dates = matrices['all_dates']
                
                # Lọc các ngày nằm trong khoảng backtest
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date)
                backtest_dates = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]
                
                progress_bar_bt = st.progress(0)
                status_bt = st.empty()
                total_days = len(backtest_dates)
                
                # Bộ nhớ đệm 60 phiên
                historical_features_window = []
                
                # Nạp sẵn (Warm-up) 60 phiên trước ngày bắt đầu Backtest
                if total_days > 0:
                    first_target_idx = all_dates.get_loc(backtest_dates[0])
                    warmup_start = max(0, first_target_idx - 60)
                    warmup_dates = all_dates[warmup_start:first_target_idx]
                    for d in warmup_dates:
                        df_d = build_daily_features(matrices, d)
                        if df_d is not None and len(df_d) > 30:
                            historical_features_window.append(df_d[['Z_Volatility', 'Market_Breadth', 'Liquidity_Drying', 'Return_Dispersion']].copy())
                
                # Chạy Backtest chính
                for i, current_date in enumerate(backtest_dates):
                    df_daily = build_daily_features(matrices, current_date)
                    
                    if df_daily is not None and len(df_daily) > 30:
                        df_flagged, heat_rate = run_ml_detection(df_daily, historical_features_window, threshold_percentile=90)
                        
                        # Cập nhật bộ nhớ trượt
                        X_today = df_daily[['Z_Volatility', 'Market_Breadth', 'Liquidity_Drying', 'Return_Dispersion']].copy()
                        historical_features_window.append(X_today)
                        if len(historical_features_window) > 60:
                            historical_features_window.pop(0)
                            
                        counts = df_flagged['Cap'].value_counts() if not df_flagged.empty else {}
                        
                        record = {
                            'Date': current_date,
                            'Heat_Rate': heat_rate,
                            'Market_Breadth': df_daily['Market_Breadth'].iloc[0] * 100,
                            'Large_Cap_Flagged': counts.get('Large', 0),
                            'Total_Flagged': len(df_flagged)
                        }
                        results_history.append(record)
                        
                    progress = min((i + 1) / total_days, 1.0)
                    progress_bar_bt.progress(progress)
                    status_bt.markdown(f"Đang phân tích: {current_date.strftime('%Y-%m-%d')} | (Ma trận Vector)")
                
                progress_bar_bt.empty()
                status_bt.empty()
                
                df_res = pd.DataFrame(results_history)
                
            if not df_res.empty:
                df_res[f'Heat_Rate_EMA_{ema_span}'] = df_res['Heat_Rate'].ewm(span=ema_span, adjust=False).mean()
                
                df_res['Large_Cap_Ratio'] = (df_res['Large_Cap_Flagged'] / df_res['Total_Flagged'].replace(0, 1)) * 100
                df_res.loc[df_res['Total_Flagged'] == 0, 'Large_Cap_Ratio'] = 0 
                df_res[f'Ratio_EMA_{ema_span}'] = df_res['Large_Cap_Ratio'].ewm(span=ema_span, adjust=False).mean()
                
                st.success("Phân tích hoàn tất! Tốc độ cải thiện gấp hàng chục lần nhờ Ma trận Dữ liệu.")
                
                # ==========================================
                # 1. BIỂU ĐỒ HEAT RATE & MARKET BREADTH
                # ==========================================
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_res['Date'], y=df_res['Heat_Rate'], mode='lines', name='Raw Heat Rate', line=dict(color='black', dash='dash', width=1)))
                fig1.add_trace(go.Scatter(x=df_res['Date'], y=df_res[f'Heat_Rate_EMA_{ema_span}'], mode='lines', name=f'Heat Rate (EMA {ema_span})', line=dict(color='#ff4b4b', width=3)))
                
                fig1.add_hline(y=35.0, line_width=2, line_dash="dash", line_color="red", annotation_text="Ngưỡng Trên (Danger Zone: >35.0)", annotation_position="top left", annotation_font=dict(color="red", size=12))
                max_heat = max(df_res['Heat_Rate'].max(), 50) if not df_res.empty else 50
                fig1.add_hrect(y0=35.0, y1=max_heat, line_width=0, fillcolor="red", opacity=0.1)

                fig1.add_hline(y=15.0, line_width=1.5, line_dash="dot", line_color="orange", annotation_text="Cảnh báo (Warning: >15.0)", annotation_position="bottom left", annotation_font=dict(color="orange", size=12))

                fig1.add_hline(y=10.0, line_width=2, line_dash="dash", line_color="green", annotation_text="Ngưỡng Dưới (Safe Zone: <10.0)", annotation_position="bottom left", annotation_font=dict(color="green", size=12))
                fig1.add_hrect(y0=0, y1=10.0, line_width=0, fillcolor="green", opacity=0.1)
                
                if 'Market_Breadth' in df_res.columns:
                    fig1.add_trace(go.Scatter(x=df_res['Date'], y=df_res['Market_Breadth'], mode='lines', name='Market Breadth (% > MA50)', yaxis='y2', line=dict(color='#1f77b4', width=2)))
                    
                fig1.update_layout(
                    title=dict(text=f"1. Rủi ro Đảo chiều Vĩ mô (EMA {ema_span} phiên)", font=dict(color="black", size=18)),
                    yaxis=dict(title=dict(text="Heat Rate (%)", font=dict(color="black")), tickfont=dict(color="black")),
                    yaxis2=dict(title=dict(text="Market Breadth (%)", font=dict(color="black")), tickfont=dict(color="black"), overlaying="y", side="right"),
                    xaxis=dict(tickfont=dict(color="black")),
                    hovermode="x unified", height=450, margin=dict(b=20),
                    template="plotly_white", paper_bgcolor='white', plot_bgcolor='white', font=dict(color="black"), legend=dict(font=dict(color="black"))
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # ==========================================
                # 2. BIỂU ĐỒ TỶ LỆ LARGE-CAP / FLAGGED
                # ==========================================
                if 'Large_Cap_Ratio' in df_res.columns:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=df_res['Date'], y=df_res['Large_Cap_Ratio'], name='Tỉ lệ ngày lẻ', marker_color='rgba(156, 114, 186, 0.4)'))
                    fig2.add_trace(go.Scatter(x=df_res['Date'], y=df_res[f'Ratio_EMA_{ema_span}'], mode='lines', name=f'Xu hướng cấu trúc (EMA {ema_span})', line=dict(color='#6a0dad', width=3)))
                    
                    fig2.add_hline(y=80, line_width=2, line_dash="solid", line_color="red", annotation_text="> 80%: Dòng tiền lớn tháo chạy", annotation_font=dict(color="red"))
                    fig2.add_hline(y=61.5, line_width=1.5, line_dash="dash", line_color="black", annotation_text="61.5%: Cân bằng tự nhiên", annotation_font=dict(color="black"))
                    fig2.add_hline(y=45, line_width=2, line_dash="solid", line_color="blue", annotation_text="< 45%: Đầu cơ sụp đổ", annotation_position="bottom right", annotation_font=dict(color="blue"))
                    
                    fig2.update_layout(
                        title=dict(text="2. Định vị Nguồn gốc Rủi ro (Large-cap vs Mid/Penny)", font=dict(color="black", size=18)),
                        yaxis=dict(title=dict(text="Tỉ lệ Large-cap / Flagged (%)", font=dict(color="black")), tickfont=dict(color="black"), range=[0, 105]),
                        xaxis=dict(tickfont=dict(color="black")),
                        hovermode="x unified", height=400,
                        template="plotly_white", paper_bgcolor='white', plot_bgcolor='white', font=dict(color="black"), legend=dict(font=dict(color="black"))
                    )

                    st.plotly_chart(fig2, use_container_width=True)


