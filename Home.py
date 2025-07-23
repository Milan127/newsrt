import streamlit as st
from utils import run_strategy

st.set_page_config(page_title="RSI + Ratio Strategy", layout="wide")
st.title("📈 RSI + Ratio Strategy Tradebook")

run_strategy()
