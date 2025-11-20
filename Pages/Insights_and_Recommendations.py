import streamlit as st

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Insights & Recommendations",
    page_icon="📊",
    layout="wide"
)

# --------------------------
# Page Title
# --------------------------
st.title("📊 PhonePe Pulse — Insights & Recommendations")
st.markdown("""
Welcome to the **Insights & Recommendations Dashboard**, derived from the **full EDA** and **50 SQL Queries** performed on the PhonePe Pulse dataset.

Below are the **business insights**, **patterns**, and **data-driven recommendations** categorized into meaningful sections.
""")

# --------------------------
# Metric Cards (Top Summary)
# --------------------------
st.subheader("📌 Overall Digital Payment Trends (Summary)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📈 Digital Transactions Growth", "40% – 60% YoY", "Strong")
with col2:
    st.metric("👥 New User Growth", "25% – 35% YoY", "Stable")
with col3:
    st.metric("🏪 Merchant Payments", "Growing Faster than P2P", "High Potential")

st.markdown("---")

# --------------------------
# 1. India-Level Insights
# --------------------------
with st.expander("🇮🇳 1. India-Level Key Insights", expanded=True):
    st.markdown("""
### **1.1 Explosive Digital Payment Growth**
- Total transactions and amount increase every year.
- Post-2020 period shows the highest surge (45%+ YoY).

### **1.2 P2P is the Largest Payment Type**
- P2P accounts for **55–70%** of total digital payment value.

### **1.3 Wallet Load & Financial Services Rising**
- Financial services segment grows at **25–30% YoY**.
- Tier-2 & Tier-3 states are driving new user adoption.
""")

# --------------------------
# 2. State-Level Insights
# --------------------------
with st.expander("🗺️ 2. State-Level Insights"):
    st.markdown("""
### **2.1 Top Performing States (Highest Amount)**  
- Maharashtra  
- Karnataka  
- Telangana  
- Tamil Nadu  
- Uttar Pradesh  

### **2.2 Highest YoY Growth States**  
- Odisha  
- Assam  
- Meghalaya  
- Telangana  
- Bihar  

### **2.3 Slow Growth / Declining States**  
- Ladakh  
- J&K  
- Lakshadweep  
- Andaman & Nicobar Islands  
""")

# --------------------------
# 3. District Insights
# --------------------------
with st.expander("🏙️ 3. District-Level Insights"):
    st.markdown("""
### **3.1 Top Districts by Transaction Amount**
- Bengaluru Urban (No. 1 in India)
- Mumbai
- Pune
- Hyderabad
- Chennai
- Lucknow

### **3.2 Fastest-Growing Districts**
- Surat  
- Nagpur  
- Indore  
- Jaipur  
- Coimbatore  
- Kochi  

### **3.3 District Disparities**
- Remote districts have high user count but **low transaction utilization**.
""")

# --------------------------
# 4. User Behavior Insights
# --------------------------
with st.expander("👥 4. User Behavior Insights"):
    st.markdown("""
### **4.1 Brand Adoption**
Top 5 brands:
- Xiaomi
- Samsung
- Vivo
- Realme  
(Apple only in metros)

### **4.2 Usage vs Onboarding Gap**
- Many states show high new user count but **low transaction activation**.

### **4.3 Seasonality**
- Q4 (festive season) spikes by **20–30%**.
""")

# --------------------------
# 5. Merchant Insights
# --------------------------
with st.expander("🏪 5. Merchant & P2M Insights"):
    st.markdown("""
### **5.1 Merchant Payments Rising Faster**
Especially strong in:
- Karnataka  
- Maharashtra  
- Telangana  
- Delhi NCR  

### **5.2 District Contribution Patterns**
Urban districts = **70–80%** of P2M value.

### **5.3 Rural Trends**
- QR adoption rising
- Average ticket size still low
""")

# --------------------------
# 6. Transaction Type Insights
# --------------------------
with st.expander("🔄 6. Transaction Type Insights"):
    st.markdown("""
### Recharge & Bill Pay
- High in UP, Bihar, Rajasthan.

### P2P Transfers Highest In
- Karnataka, Maharashtra, Telangana.

### Financial Services Growing In
- Tamil Nadu  
- Kerala  
- Andhra Pradesh  
""")

# --------------------------
# 7. Deep Hidden Patterns
# --------------------------
with st.expander("🧠 7. Hidden Patterns & Advanced Insights"):
    st.markdown("""
### **Insight 1 — High user growth ≠ high transaction value**  
Several states show strong user onboarding but weak activation.

### **Insight 2 — Metro districts behave like separate countries**  
Bengaluru Urban > entire Northeast in value.

### **Insight 3 — Cultural patterns in Transaction Type**  
North, South, East show different payment behaviour clusters.

### **Insight 4 — Urbanization strongly correlates with digital payments**  
High migration → higher digital adoption.

### **Insight 5 — Small states exhibit false YoY spikes**  
Base values very low → inflated percentages.
""")

# --------------------------
# 8. Recommendations
# --------------------------
with st.expander("🎯 8. Recommendations"):
    st.markdown("""
## **A. For Product/Growth Teams**
- Push Merchant Payments in Tier-2 cities.
- Increase user activation (not just onboarding).
- Run incentives in slower-growth states.

## **B. For Operations**
- Prioritize QR deployment where users > transactions.
- Improve auto-pay adoption.
- Support merchant onboarding drives.

## **C. For Engineering/Data**
- Build anomaly detection for state-level dips.
- Create district segmentation (urban/semi-urban/rural).
- Strengthen fraud detection in high-value states.

## **D. For Marketing**
- Metro vs rural campaigns must be separate.
- Promote P2P cashback in high-traffic states.
- Promote bill-pay cashback in prepaid-heavy states.

## **E. For Strategy / Leadership**
- Biggest opportunity:  
  **Tier-2 merchant ecosystem + bill payment penetration**.
""")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("Generated from EDA & 50 SQL Queries | PhonePe Pulse Data Analytics")