import streamlit as st
from utils import display_chart

st.set_page_config(page_title="Chart", layout="wide")
st.title("📈 Price Chart")

display_chart()
