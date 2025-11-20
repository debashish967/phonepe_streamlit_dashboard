# PhonePe Pulse Dashboard (Streamlit)

A fully interactive Streamlit-based dashboard replicating the core functionality and design philosophy of the **PhonePe Pulse Dashboard**.  
This project includes **3D maps**, **state & district visualizations**, **user and brand insights**, **SQL analytics (50 queries)**, and an **Insights & Recommendations** page.

---

## 🚀 Features

### ✅ Home Page
- 2D choropleth map of India  
- 3D map using PyDeck (extruded polygons + bar columns)  
- Year/Quarter/Transaction Type filters  
- KPIs showing:
  - Total Transaction Amount  
  - Total Transaction Count  
  - Average Transaction Value  

---

## 📊 State Analysis Page
- Year-wise and quarter-wise comparison  
- Line charts & bar charts  
- Top-performing states  
- State-level transaction trends  

---

## 🗺️ District Analysis Page
- District-level bubbles and bar charts  
- District comparison of transaction count & amount  

---

## 👥 User & Brand Analysis Page
- Android vs iOS user distribution  
- Smartphone brand share  
- User adoption patterns  

---

## 🧮 SQL Insights (50 Queries)
- All 50 SQL queries integrated with filters  
- Dynamic tables with sorting (ascending/descending)  
- Powered by SQLite database `phonepe.db`

---

## 💡 Insights & Recommendations
- Automatically generated business insights  
- Growth opportunities  
- State & district performance analysis  
- User adoption strategy recommendations  

---

## 🗂️ Project Structure

```
phonepe_streamlit_dashboard/
│── app.py                         # Main Streamlit app  
│── pages/
│    └── Insights_and_Recommendations.py
│── phonepe.db                     # SQLite database  
│── india_states.geojson           # GeoJSON for state boundaries  
│── README.md                      # Project documentation  
│── requirements.txt               # Python dependencies  
│── .gitignore                     # Git ignore patterns  
```

---

## 🔧 Installation

### 1️⃣ Install dependencies  
```
pip install -r requirements.txt
```

### 2️⃣ Run the app  
```
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push the entire folder to GitHub  
2. Go to https://share.streamlit.io  
3. Choose your repository  
4. Set **app.py** as entry file  
5. Deploy 🚀  

---

## 📝 Notes
- The GeoJSON file must be stored locally as **india_states.geojson**  
- The project uses **PyDeck** for 3D visualization  
- A large GeoJSON is optimized for smooth performance  

---

## 🤝 Contributions
Feel free to open issues or pull requests.

---

## 📜 License
MIT License

---

## 🧑‍💻 Developed By

### Debashish Borah
Designed & Built with ❤️ using Python, Streamlit & SQLite