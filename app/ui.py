import streamlit as st
import pandas as pd
import requests


st.set_page_config(page_title="LogiRisk Dashboard", layout="wide", page_icon="🚚")


API_URL = "https://logirisk-api-716133749292.us-central1.run.app/predict"
# API_URL = "http://127.0.0.1:8000/predict"

RAW_COLUMNS = [
    'type', 'days_for_shipment_scheduled', 'category_id', 'customer_segment',
    'department_id', 'market', 'order_item_quantity', 'product_price',
    'shipping_mode', 'order_city', 'order_date'
]


st.title("🚚 LogiRisk ML")
st.markdown("Analyze supply chain risk scores in real-time.")

uploaded_file = st.file_uploader("Upload Raw Logistics CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    
    if missing:
        st.error(f"❌ Your CSV is missing: {missing}")
    else:
        if st.button("Start Audit"):
            scores = []
            status = []
            progress = st.progress(0)
            
            for i, row in df[RAW_COLUMNS].iterrows():
                try:
                    r = requests.post(API_URL, json=row.to_dict(), timeout=5)
                    data = r.json()

                    scores.append(data.get("risk_score", 0.0))
                    status.append("High Risk" if data.get("late_risk") else "Low Risk")
                except:
                    scores.append(0.0)
                    status.append("API Error")
                
                progress.progress((i + 1) / len(df))
            
            df['Risk_Score'] = scores
            df['Late_Risk'] = status

            st.divider()
            st.subheader("📋 Audit Summary")
            
            c1, c2, c3, c4 = st.columns(4)
            total_count = len(df)
            high_risk_count = (df['Late_Risk'] == "High Risk").sum()
            avg_score = df['Risk_Score'].mean()
            value_at_risk = df[df['Late_Risk'] == "High Risk"]['product_price'].sum()

            with c1:
                st.metric("Total Audited", total_count)
            with c2:
                st.metric("High Risk Orders", high_risk_count, f"{high_risk_count/total_count:.1%}", delta_color="inverse")
            with c3:
                st.metric("Avg Risk Score", f"{avg_score:.2f}")
            with c4:
                st.metric("Revenue at Risk", f"${value_at_risk:,.2f}")


            st.subheader("🔍 Detailed Shipment Logs")
            
            def color_risk(val):
                if val == "High Risk": return 'background-color: #ff4b4b; color: white'
                if val == "Low Risk": return 'background-color: #28a745; color: white'
                return ''

            st.dataframe(df.style.applymap(color_risk, subset=['Late_Risk']), use_container_width=True)
            
            st.download_button("📥 Download Full Risk Report", df.to_csv(index=False), "risk_audit_report.csv")