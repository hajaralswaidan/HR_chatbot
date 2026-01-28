import streamlit as st
import pandas as pd
from pathlib import Path # for path

#page-------------
st.set_page_config(page_title="HR Chatbot", layout="wide")
st.title(" HR Dataset Check")

#path for data base----
DATA_PATH = Path("data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(DATA_PATH)


if not DATA_PATH.exists():
    st.error("CSV file not found in the data/ folder.")
    st.info("Put the file here: data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    st.stop()

# read pandas---

#st.success("CSV loaded successfully")
#st.write("Rows:", len(df))
#st.write("Columns:", df.shape[1])
#st.write("Column names:", list(df.columns))

#test  ---------------------

if "messages" not in st.session_state:
    st.session_state.messages= []

for msg in st.session_state.messages:
    st.write(msg)


user_input= st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append("You: " + user_input)
    st.rerun()