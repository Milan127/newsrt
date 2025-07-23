import streamlit as st
from utils import display_trade_log

st.set_page_config(page_title="Trade Log", layout="wide")
st.title("📑 Trade Log")

display_trade_log()
