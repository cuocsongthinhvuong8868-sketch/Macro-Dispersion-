import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import os
from vnstock import Quote

# ==============================================================================
# 1. CẤU HÌNH & GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Macro Dispersion & Delta Bootstrapping", layout="wide")
os.environ['VNSTOCK_API_KEY'] = "vnstock_17b56a86b930db526e25e8de447a0bfd"

st.sidebar.title("⚙️ Cấu hình Mô phỏng")

st.sidebar.markdown("### 1. Dữ liệu lịch sử")
history_years = st.sidebar.number_input("Số năm tải dữ liệu gốc (Tối thiểu 3 năm):", min_value=3, max_value=20, value=5, step=1)

st.sidebar.markdown("### 2. Tham số Bootstrapping")
n_days = st.sidebar.number_input("Số phiên dự phóng (N):", min_value=5, max_value=120, value=20, step=1)
target_return = st.sidebar.number_input("Mức tăng kỳ vọng X (%):", min_value=-20.0, max_value=50.0, value=5.0, step=0.5)

st.sidebar.markdown("### 3. Tùy chọn thời gian chạy MC & Biểu đồ")
mc_start_date = st.sidebar.date_input("Từ ngày:", value=datetime.date.today() - datetime.timedelta(days=252))
mc_end_date = st.sidebar.date_input("Đến ngày:", value=datetime.date.today())

st.sidebar.markdown("### 4. Tham số Động lượng & Thresholds")
delta_steps = st.sidebar.slider("Bước nhảy Delta (VD: 5 = Động lượng 1 tuần):", min_value=1, max_value=20, value=5, step=1)
delta_window = st.sidebar.number_input("Cửa sổ MA cho Delta (Ngày):", min_value=20, max_value=120, value=60, step=1)
mc_window = st.sidebar.number_input("Cửa sổ Rolling Volatility cho MC:", min_value=20, max_value=120, value=60, step=1)

n_sims_threshold = 1000 

# ==============================================================================
# 2. HÀM TẢI & XỬ LÝ DỮ LIỆU (NGUỒN KBS)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_market_data(years):
    try:
        df_tickers = pd.read_csv("danh_sach_200_ma.csv")
        tickers = df_tickers['Ticker'].tolist()
    except Exception as e:
        st.error(f"Không tìm thấy file danh_sach_150_ma.csv. Lỗi: {e}")
        return None, None

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(years * 365.25))
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    progress_bar = st.progress(0)
    status_text = st.empty()
    data_dict = {}
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Đang tải: {ticker} ({i+1}/{len(tickers)})")
        try:
            q = Quote(symbol=ticker, source='KBS')
            df_hist = q.history(start=start_str, end=end_str)
            if df_hist is not None and not df_hist.empty:
                df_hist.columns = [str(col).lower() for col in df_hist.columns]
                if 'close' in df_hist.columns:
                    df_hist['time'] = pd.to_datetime(df_hist['time']).dt.normalize()
                    df_hist = df_hist.set_index('time')
                    df_hist = df_hist[~df_hist.index.duplicated(keep='last')]
                    data_dict[ticker] = df_hist['close']
        except:
            pass
        
        progress_bar.progress((i + 1) / (len(tickers) + 1))
        time.sleep(1.1) 

    status_text.text("Đang tải chỉ số VN-INDEX (Nguồn KBS)...")
    vnindex_series = None
    try:
        q_vni = Quote(symbol='VNINDEX', source='KBS') 
        df_vni = q_vni.history(start=start_str, end=end_str)
        if df_vni is not None and not df_vni.empty:
            df_vni.columns = [str(col).lower() for col in df_vni.columns]
            df_vni['time'] = pd.to_datetime(df_vni['time']).dt.normalize()
            df_vni = df_vni.set_index('time')
            vnindex_series = df_vni['close']
    except:
        pass
        
    time.sleep(1.1) 
    progress_bar.empty()
    status_text.empty()

    if not data_dict: return None, None
    if vnindex_series is None or vnindex_series.empty:
        st.error("❌ LỖI: Không thể tải VN-Index gốc từ KBS. Hệ thống từ chối việc nội suy giả tạo.")
        return None, None

    df_prices = pd.DataFrame(data_dict).ffill().dropna(how='all')
    df_prices = df_prices.reindex(vnindex_series.index).ffill()
    
    return df_prices, vnindex_series

# ==============================================================================
# 3. LÕI TOÁN HỌC: DISPERSION, MC THRESHOLDS, DELTA & BOOTSTRAPPING
# ==============================================================================
def calculate_dispersion_and_delta(df_prices, vnindex_series, delta_steps=1):
    stock_returns = df_prices.pct_change().dropna(how='all')
    market_returns = vnindex_series.pct_change().reindex(stock_returns.index).ffill()
    deviations = stock_returns.sub(market_returns, axis=0)
    
    annualize_factor = np.sqrt(252)
    
    csad = deviations.abs().mean(axis=1) * annualize_factor
    cssd = np.sqrt((deviations**2).sum(axis=1) / (stock_returns.shape[1] - 1)) * annualize_factor
    spread = cssd - csad
    
    delta_cssd = cssd.diff(periods=delta_steps)
    delta_csad = csad.diff(periods=delta_steps)
    delta_spread = spread.diff(periods=delta_steps)
    
    return stock_returns, market_returns, cssd, csad, spread, delta_cssd, delta_csad, delta_spread

def calculate_mc_thresholds_dual(stock_returns, window=60, n_sims=1000):
    rolling_vol = stock_returns.rolling(window=window, min_periods=window//2).std()
    
    cssd_upper, cssd_lower, csad_upper, csad_lower = [pd.Series(index=stock_returns.index, dtype=float) for _ in range(4)]
    num_stocks = stock_returns.shape[1]
    annualize_factor = np.sqrt(252)
    
    for i in range(window, len(stock_returns)):
        daily_vols = rolling_vol.iloc[i].fillna(0).values
        if np.sum(daily_vols) == 0: continue
            
        sim_returns = np.random.normal(0, daily_vols, size=(n_sims, num_stocks))
        sim_market_returns = np.mean(sim_returns, axis=1, keepdims=True)
        
        sim_cssd = np.sqrt(np.sum((sim_returns - sim_market_returns)**2, axis=1) / (num_stocks - 1)) * annualize_factor
        sim_csad = np.mean(np.abs(sim_returns - sim_market_returns), axis=1) * annualize_factor
        
        cssd_upper.iloc[i], cssd_lower.iloc[i] = np.percentile(sim_cssd, 95), np.percentile(sim_cssd, 5)
        csad_upper.iloc[i], csad_lower.iloc[i] = np.percentile(sim_csad, 95), np.percentile(sim_csad, 5)
        
    return (cssd_upper.ewm(span=5).mean(), cssd_lower.ewm(span=5).mean(), 
            csad_upper.ewm(span=5).mean(), csad_lower.ewm(span=5).mean())

def calculate_rolling_bootstrapping_by_delta(market_returns, delta_spread, window_std, n_days, target_return_pct, start_date, end_date, lookback_pool=252):
    results = []
    target_decimal = target_return_pct / 100.0
    
    mask = (market_returns.index.date >= start_date) & (market_returns.index.date <= end_date)
    dates_to_compute = market_returns.index[mask]
    
    if len(dates_to_compute) == 0:
        return pd.DataFrame(columns=['Date', 'Prob_Win', 'Exp_Ret']).set_index('Date')
    
    rolling_std_delta = delta_spread.rolling(window=window_std).std()
    is_normal_mask = delta_spread.abs() <= rolling_std_delta
    
    for t in dates_to_compute:
        t_loc = market_returns.index.get_loc(t)
        start_idx = max(0, t_loc - lookback_pool)
        
        past_returns = market_returns.iloc[start_idx : t_loc + 1]
        past_mask = is_normal_mask.iloc[start_idx : t_loc + 1]
        
        normal_pool = past_returns[past_mask].dropna().values
        
        if len(normal_pool) < 30:
            results.append((t, np.nan, np.nan))
            continue
            
        picks = np.random.choice(normal_pool, size=(3000, n_days), replace=True)
        paths = np.prod(1 + picks, axis=1) - 1
        
        prob = np.mean(paths >= target_decimal) * 100
        exp_r = np.mean(paths) * 100
        results.append((t, prob, exp_r))
        
    return pd.DataFrame(results, columns=['Date', 'Prob_Win', 'Exp_Ret']).set_index('Date')

# ==============================================================================
# 4. CHẠY CHƯƠNG TRÌNH & VẼ BIỂU ĐỒ
# ==============================================================================
st.title("Macro Dispersion & Delta Bootstrapping Radar")

if st.sidebar.button("🚀 Bắt Đầu Tính Toán"):
    if mc_start_date > mc_end_date:
        st.error("❌ LỖI: 'Từ ngày' không thể lớn hơn 'Đến ngày'. Vui lòng chọn lại.")
    else:
        with st.spinner(f"Đang xử lý Toán học Động lượng và Mô phỏng MC từ {mc_start_date} đến {mc_end_date}..."):
            df_prices, vnindex_series = load_market_data(history_years)
            
            if df_prices is not None and vnindex_series is not None:
                st_ret, mkt_ret, cssd, csad, spread, d_cssd, d_csad, d_spread = calculate_dispersion_and_delta(df_prices, vnindex_series, delta_steps)
                cssd_upper, cssd_lower, csad_upper, csad_lower = calculate_mc_thresholds_dual(st_ret, window=mc_window, n_sims=n_sims_threshold)
                
                # Gộp các đường upper/lower vào chung df_plot ĐỂ ĐỒNG BỘ ĐỘ DÀI KHI DROPNA
                df_plot = pd.DataFrame({
                    'CSSD': cssd * 100, 'CSAD': csad * 100, 'Spread': spread * 100,
                    'Delta_CSSD': d_cssd * 100, 'Delta_CSAD': d_csad * 100, 'Delta_Spread': d_spread * 100,
                    'CSSD_Upper': cssd_upper * 100, 'CSSD_Lower': cssd_lower * 100,
                    'CSAD_Upper': csad_upper * 100, 'CSAD_Lower': csad_lower * 100,
                    'VNINDEX': vnindex_series
                }).dropna()

                df_rolling_mc = calculate_rolling_bootstrapping_by_delta(
                    mkt_ret, d_spread, window_std=delta_window, n_days=n_days, 
                    target_return_pct=target_return, start_date=mc_start_date, end_date=mc_end_date
                )

                if df_rolling_mc.empty:
                    st.warning(f"⚠️ Không có dữ liệu giao dịch nào trong khoảng thời gian từ {mc_start_date} đến {mc_end_date}.")
                else:
                    latest_prob = df_rolling_mc['Prob_Win'].dropna().iloc[-1] if not df_rolling_mc['Prob_Win'].dropna().empty else 0
                    latest_exp = df_rolling_mc['Exp_Ret'].dropna().iloc[-1] if not df_rolling_mc['Exp_Ret'].dropna().empty else 0
                    
                    st.markdown("---")
                    st.subheader(f"🎯 Snapshot Cuối Cùng (Dự phóng {n_days} phiên)")
                    st.markdown(f"*Dựa trên ngày giao dịch cuối cùng trong khoảng thời gian bạn chọn.*")
                    
                    col1, col2 = st.columns(2)
                    col1.metric(f"Xác suất VNI tăng > {target_return}%", f"{latest_prob:.2f} %")
                    col2.metric(f"Lợi suất Kỳ vọng (Expected Return)", f"{latest_exp:.2f} %")

                    # === VẼ BIỂU ĐỒ 1: ĐỘNG LƯỢNG XÁC SUẤT ===
                    st.markdown("---")
                    st.subheader(f"📈 1. Lịch sử Giả Lập MC về X % return on N days ")
                    
                    fig_mc = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_mc.add_trace(go.Scatter(x=df_rolling_mc.index, y=df_rolling_mc['Prob_Win'], name='Win Rate (%)', line=dict(color='#0066cc', width=2)), secondary_y=False)
                    fig_mc.add_trace(go.Scatter(x=df_rolling_mc.index, y=df_rolling_mc['Exp_Ret'], name='Expected Return (%)', line=dict(color='#cc0000', width=2)), secondary_y=True)
                    fig_mc.add_hline(y=0, line_dash="solid", line_color="black", secondary_y=True)

                    fig_mc.update_layout(height=450, template='plotly_white', hovermode='x unified')
                    fig_mc.update_xaxes(title_text="Thời gian")
                    fig_mc.update_yaxes(title_text="Win Rate (%)", secondary_y=False, color='#0066cc')
                    fig_mc.update_yaxes(title_text="Expected Return (%)", secondary_y=True, color='#cc0000')
                    st.plotly_chart(fig_mc, use_container_width=True)

                    # === VẼ BIỂU ĐỒ 2: HỆ THỐNG PHÂN RÃ ===
                    st.markdown("---")
                    st.subheader(f"📊 2. Phân rã Vĩ mô & Vận tốc Tâm lý ({mc_start_date.strftime('%d/%m/%Y')} - {mc_end_date.strftime('%d/%m/%Y')})")
                    st.markdown("*Biểu đồ 1 & 2: Kèm vận tốc Delta. Biểu đồ 3: Spread thuần túy.*")
                    
                    # Lọc biểu đồ phân rã theo thời gian đã chọn (đồng bộ hoàn toàn)
                    plot_mask = (df_plot.index.date >= mc_start_date) & (df_plot.index.date <= mc_end_date)
                    df_plot_filtered = df_plot.loc[plot_mask]

                    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                                        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]],
                                        subplot_titles=(
                                            "CSSD (Stdev) kẹp giữa Ngưỡng MC & Gia tốc Delta", 
                                            "CSAD (Absolute) kẹp giữa Ngưỡng MC & Gia tốc Delta", 
                                            "Spread (CSSD - CSAD) Thuần túy"
                                        ))

                    # Sử dụng trực tiếp các cột Upper/Lower từ bảng đã gộp
                    # --- CSSD ---
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['CSSD_Upper'], name='CSSD 95% Upper', line=dict(color='orange', dash='dash')), row=1, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['CSSD_Lower'], name='CSSD 5% Lower', line=dict(color='green', dash='dash')), row=1, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['CSSD'], name='CSSD Thực tế', line=dict(color='red', width=2)), row=1, col=1, secondary_y=False)
                    fig.add_trace(go.Bar(x=df_plot_filtered.index, y=df_plot_filtered['Delta_CSSD'], name='Delta CSSD', marker_color='rgba(255, 99, 71, 0.3)'), row=1, col=1, secondary_y=True)
                    
                    # --- CSAD ---
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['CSAD_Upper'], name='CSAD 95% Upper', line=dict(color='orange', dash='dash')), row=2, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['CSAD_Lower'], name='CSAD 5% Lower', line=dict(color='green', dash='dash')), row=2, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['CSAD'], name='CSAD Thực tế', line=dict(color='blue', width=2)), row=2, col=1, secondary_y=False)
                    fig.add_trace(go.Bar(x=df_plot_filtered.index, y=df_plot_filtered['Delta_CSAD'], name='Delta CSAD', marker_color='rgba(65, 105, 225, 0.3)'), row=2, col=1, secondary_y=True)

                    # --- SPREAD (THUẦN TÚY) ---
                    fig.add_trace(go.Scatter(x=df_plot_filtered.index, y=df_plot_filtered['Spread'], name='Raw Spread', line=dict(color='purple', width=2)), row=3, col=1)
                    fig.add_hline(y=0, line_dash="solid", line_color="black", row=3, col=1)

                    fig.update_xaxes(showticklabels=True)
                    fig.update_layout(height=1300, template='plotly_white', hovermode='x unified', barmode='relative')
                    
                    for i in range(1, 4):
                        fig.update_yaxes(title_text="Độ phân tán (Annualized %)", row=i, col=1, secondary_y=False)
                        if i < 3:
                            fig.update_yaxes(title_text="Delta (%)", showgrid=False, row=i, col=1, secondary_y=True)

                    st.plotly_chart(fig, use_container_width=True)