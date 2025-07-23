import streamlit as st
from utils import display_summary

st.set_page_config(page_title="Summary", layout="wide")
st.title("📊 Trade Summary & Performance")

display_summary()
