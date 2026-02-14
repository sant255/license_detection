import streamlit as st
import pandas as pd
import sqlite3

st.title("🚗 Intelligent License Plate Reader Dashboard")

uploaded = st.file_uploader("Upload Vehicle Image")

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())

    plates = detect_plate("temp.jpg")

    for plate in plates:
        cleaned = clean_text(plate)
        insert_plate(cleaned)
        st.success(f"Detected Plate: {cleaned}")

conn = sqlite3.connect("vehicles.db")
df = pd.read_sql_query("SELECT * FROM vehicle_logs", conn)

st.subheader("📊 Analytics")

st.bar_chart(df["plate"].value_counts())
st.line_chart(df.groupby("timestamp").count())