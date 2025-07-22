import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go

# === Settings ===
st.set_page_config(page_title="RSI + Ratio Tradebook", layout="wide")
st.title("📈 RSI + Ratio Strategy Tradebook Viewer")

# === Upload CSV ===
uploaded_file = st.file_uploader("Upload CSV with Stock Symbols (Column: 'Symbol')", type=["csv"])

# === Date Range ===
col1, col2 = st.columns(2)
def_date = pd.to_datetime("2024-10-01")
start_date = col1.date_input("Start Date", value=def_date)
end_date = col2.date_input("End Date", value=pd.to_datetime("today"))

# === Download Data ===
def download_data(symbols, start, end):
    symbols_ns = [s + ".NS" for s in symbols]
    return yf.download(symbols_ns, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)

# === Strategy Function ===
def evaluate_strategy(stock_data, stock_name):
    trades = []
    if stock_data.isnull().values.any():
        return trades

    df = pd.DataFrame(index=stock_data.index)
    df['Close'] = stock_data['Close']
    df['124DMA'] = df['Close'].rolling(window=124).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['Ratio'] = df['Close'] / df['124DMA']

    buy_price = None
    for i in range(len(df)):
        date = df.index[i]
        rsi = df.iloc[i]['rsi']
        ratio = df.iloc[i]['Ratio']
        ltp = df.iloc[i]['Close']

        if buy_price is None and rsi < 30 and ratio < 0.80:
            buy_price = ltp
            trades.append({
                "Stock": stock_name,
                "Date": date,
                "Action": "Buy",
                "Price": ltp,
                "RSI": rsi,
                "Ratio": ratio,
                "Investment": 5000,
                "Qty": round(5000 / ltp, 2)
            })

        elif buy_price is not None and (rsi > 70 or ratio > 1.3 or ltp < 0.75 * buy_price):
            sell_price = ltp
            qty = round(5000 / buy_price, 2)
            pnl = (sell_price - buy_price) * qty
            pnl_pct = ((sell_price - buy_price) / buy_price) * 100
            trades.append({
                "Stock": stock_name,
                "Date": date,
                "Action": "Sell",
                "Price": sell_price,
                "RSI": rsi,
                "Ratio": ratio,
                "P&L (\u20b9)": round(pnl, 2),
                "P&L (%)": round(pnl_pct, 2)
            })
            buy_price = None

    return trades

# === Chart Function ===
def plot_price_chart(df, trade_df):
    df = df.copy().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))

    buys = trade_df[trade_df['Action'] == 'Buy']
    sells = trade_df[trade_df['Action'] == 'Sell']

    fig.add_trace(go.Scatter(
        x=buys['Date'],
        y=buys['Price'],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Buy'))

    fig.add_trace(go.Scatter(
        x=sells['Date'],
        y=sells['Price'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Sell'))

    fig.update_layout(height=500, title="Trade Chart", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# === Run Strategy ===
if uploaded_file:
    df_symbols = pd.read_csv(uploaded_file)
    if 'Symbol' not in df_symbols.columns:
        st.error("CSV must contain 'Symbol' column.")
    else:
        symbols = df_symbols['Symbol'].dropna().unique().tolist()

        if st.button("\ud83d\udd0d Run Strategy"):
            with st.spinner("\ud83d\udcc5 Processing symbols..."):
                data = download_data(symbols, start_date, end_date)

                all_trades = []
                charts = {}
                for sym in symbols:
                    try:
                        df = data[sym + ".NS"]
                        trades = evaluate_strategy(df, sym)
                        if trades:
                            all_trades.extend(trades)
                            charts[sym] = df
                    except Exception as e:
                        st.warning(f"Error with {sym}: {e}")

                if all_trades:
                    df_trades = pd.DataFrame(all_trades)
                    df_trades.sort_values(by="Date", inplace=True)

                    sell_trades = df_trades[df_trades['Action'] == 'Sell']
                    total_profit = sell_trades['P&L (₹)'].sum()
                    avg_return = sell_trades['P&L (%)'].mean()
                    win_trades = sell_trades[sell_trades['P&L (₹)'] > 0]
                    win_rate = (len(win_trades) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0

                    st.subheader("\ud83d\udcca Trade Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Trades", len(sell_trades))
                    col2.metric("Win %", f"{win_rate:.2f}%")
                    col3.metric("Total P&L", f"₹{total_profit:,.2f}")
                    col4.metric("Avg Return", f"{avg_return:.2f}%")

                    # Filters
                    stocks = sorted(df_trades["Stock"].unique())
                    actions = ["All", "Buy", "Sell"]
                    colf1, colf2 = st.columns(2)
                    stock_filter = colf1.selectbox("Filter by Stock", ["All"] + stocks)
                    action_filter = colf2.selectbox("Filter by Action", actions)

                    df_display = df_trades.copy()
                    if stock_filter != "All":
                        df_display = df_display[df_display["Stock"] == stock_filter]
                    if action_filter != "All":
                        df_display = df_display[df_display["Action"] == action_filter]

                    st.dataframe(df_display, use_container_width=True)
                    st.download_button("\ud83d\udcc2 Download CSV", df_trades.to_csv(index=False), file_name="tradebook.csv")

                    if stock_filter != "All" and stock_filter in charts:
                        st.subheader(f"\ud83d\udcc8 Chart for {stock_filter}")
                        plot_price_chart(charts[stock_filter], df_trades[df_trades['Stock'] == stock_filter])
                else:
                    st.warning("\u26a0\ufe0f No trades generated.")
