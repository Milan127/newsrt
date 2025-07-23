import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def download_data(symbols, start, end):
    symbols_ns = [s + ".NS" for s in symbols]
    return yf.download(symbols_ns, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)

def evaluate_strategy(stock_data, stock_name, per_stock_investment):
    trades = []
    if stock_data.isnull().values.any():
        return trades
    df = pd.DataFrame(index=stock_data.index)
    df['Close'] = stock_data['Close']
    df['124DMA'] = df['Close'].rolling(window=124).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['Ratio'] = df['Close'] / df['124DMA']
    buy_price = None
    buy_qty = None
    buy_date = None
    for i in range(len(df)):
        date = df.index[i]
        rsi = df.iloc[i]['rsi']
        ratio = df.iloc[i]['Ratio']
        ltp = df.iloc[i]['Close']
        if buy_price is None and rsi < 30 and ratio < 0.80:
            buy_price = ltp
            buy_qty = int(per_stock_investment / ltp)
            buy_date = date
            trades.append({
                "Stock": stock_name,
                "Date": date,
                "Action": "Buy",
                "Price": ltp,
                "RSI": rsi,
                "Ratio": ratio,
                "Qty": buy_qty,
                "Investment": round(buy_price * buy_qty, 2),
                "P&L (₹)": None,
                "Capital": None,
                "Days Held": None
            })
        elif buy_price is not None and (rsi > 70 or ratio > 1.3 or ltp < 0.75 * buy_price):
            sell_price = ltp
            pnl = (sell_price - buy_price) * buy_qty
            days_held = (date - buy_date).days if buy_date else None
            trades.append({
                "Stock": stock_name,
                "Date": date,
                "Action": "Sell",
                "Price": sell_price,
                "RSI": rsi,
                "Ratio": ratio,
                "Qty": buy_qty,
                "Investment": None,
                "P&L (₹)": round(pnl, 2),
                "Capital": None,
                "Days Held": days_held
            })
            buy_price = None
            buy_qty = None
            buy_date = None
    return trades

def run_strategy():
    uploaded_file = st.file_uploader("Upload CSV (Column: Symbol)", type=["csv"])
    col1, col2 = st.columns(2)
    def_date = pd.to_datetime("2024-10-01")
    start_date = col1.date_input("Start Date", value=def_date)
    end_date = col2.date_input("End Date", value=pd.to_datetime("today"))
    total_capital = st.number_input("Total Capital (₹)", value=100000)
    per_stock_investment = st.number_input("Investment per Stock (₹)", value=5000)
    if uploaded_file:
        df_symbols = pd.read_csv(uploaded_file)
        if 'Symbol' not in df_symbols.columns:
            st.error("CSV must contain 'Symbol' column.")
            return
        symbols = df_symbols['Symbol'].dropna().unique().tolist()
        if st.button("⭐ Run Strategy"):
            with st.spinner("⏳ Processing..."):
                data = download_data(symbols, start_date, end_date)
                all_trades = []
                charts = {}
                for sym in symbols:
                    try:
                        df = data[sym + ".NS"]
                        trades = evaluate_strategy(df, sym, per_stock_investment)
                        if trades:
                            all_trades.extend(trades)
                            charts[sym] = df
                    except Exception as e:
                        st.warning(f"Error with {sym}: {e}")
                if all_trades:
                    df_trades = pd.DataFrame(all_trades).sort_values(by="Date")
                    st.session_state.trades = df_trades
                    st.session_state.charts = charts
                    st.success("✅ Strategy completed. Visit other tabs.")

def display_summary():
    if 'trades' not in st.session_state:
        st.warning("Please run the strategy from the Home tab.")
        return
    df_trades = st.session_state.trades.copy()
    total_capital = 100000
    capital = total_capital
    capital_progress = []
    for i, row in df_trades.iterrows():
        if row['Action'] == 'Buy':
            capital -= row['Investment']
        elif row['Action'] == 'Sell':
            capital += row['Qty'] * row['Price']
            df_trades.at[i, 'P&L (₹)'] = round((row['Price'] - df_trades.at[i - 1, 'Price']) * row['Qty'], 2)
        df_trades.at[i, 'Capital'] = round(capital, 2)
        capital_progress.append({"Date": row['Date'], "Capital": capital})
    st.session_state.trades = df_trades
    sell_trades = df_trades[df_trades['Action'] == 'Sell']
    total_profit = sell_trades['P&L (₹)'].sum()
    avg_return = sell_trades['P&L (₹)'].mean()
    win_trades = sell_trades[sell_trades['P&L (₹)'] > 0]
    win_rate = (len(win_trades) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", len(sell_trades))
    col2.metric("Win %", f"{win_rate:.2f}%")
    col3.metric("Total P&L", f"₹{total_profit:,.2f}")
    col4.metric("Avg P&L", f"₹{avg_return:.2f}")
    leaderboard = sell_trades.groupby("Stock")["P&L (₹)"].sum().sort_values(ascending=False).head(5)
    st.subheader("🏆 Top 5 Stocks by P&L")
    st.dataframe(leaderboard.reset_index(), use_container_width=True)
    cap_df = pd.DataFrame(capital_progress).dropna()
    cap_chart = go.Figure()
    cap_chart.add_trace(go.Scatter(x=cap_df['Date'], y=cap_df['Capital'], mode='lines+markers', name='Capital'))
    cap_chart.update_layout(title="Capital Growth Over Time", xaxis_title="Date", yaxis_title="Capital (₹)", template="plotly_white")
    st.plotly_chart(cap_chart, use_container_width=True)

def display_trade_log():
    if 'trades' not in st.session_state:
        st.warning("Please run the strategy from the Home tab.")
        return
    df_trades = st.session_state.trades.copy()
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
    st.dataframe(df_display, use_container_width=True, height=600)
    st.download_button("📂 Download CSV", df_trades.to_csv(index=False), file_name="tradebook.csv")
    with st.expander("🧾 View Full Trade Log"):
        log_output = "
".join([f"{t['Date']} | {t['Stock']} | {t['Action']} | {t['Price']} | Qty: {t['Qty']}" for _, t in df_trades.iterrows()])
        st.text_area("Trade Log", log_output, height=300)

def display_chart():
    if 'trades' not in st.session_state or 'charts' not in st.session_state:
        st.warning("Please run the strategy from the Home tab.")
        return
    df_trades = st.session_state.trades
    charts = st.session_state.charts
    stocks = sorted(df_trades["Stock"].unique())
    stock_filter = st.selectbox("Select Stock for Chart", stocks)
    if stock_filter not in charts:
        st.warning("No chart available.")
        return
    df = charts[stock_filter]
    df = df.copy().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
    buys = df_trades[(df_trades['Stock'] == stock_filter) & (df_trades['Action'] == 'Buy')]
    sells = df_trades[(df_trades['Stock'] == stock_filter) & (df_trades['Action'] == 'Sell')]
    fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
    fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))
    fig.update_layout(height=500, title="Trade Chart", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
