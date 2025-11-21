import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# -------------------------------------------------
# SINGLE CORRECT PAGE CONFIG (as you requested)
# -------------------------------------------------
import streamlit as st
from PIL import Image

logo_path = "images.jpeg"
page_icon = "📱"

if os.path.exists(logo_path):
    try:
        logo_img = Image.open(logo_path)
        page_icon = logo_img
    except:
        pass

st.set_page_config(
    page_title="PhonePe Pulse — Dashboard & Insights",
    page_icon=page_icon,
    layout="wide"
)
# -------------------------------------------------

import json
import sqlite3
from math import fsum
import pandas as pd
import numpy as np
import plotly.express as px

# REMOVE duplicate line (already set above)
# st.set_page_config(layout="wide")

st.session_state.setdefault("init", False)

# pydeck for 3D map
try:
    import pydeck as pdk
except Exception as e:
    raise RuntimeError("pydeck is required. Install with: pip install pydeck") from e

# AgGrid (optional)
USE_AGGRID = False
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    USE_AGGRID = True
except Exception:
    USE_AGGRID = False

# Shapely optional (for geometry fixes)
SHAPELY_AVAILABLE = False
try:
    from shapely.geometry import shape, mapping, Polygon, MultiPolygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False

# -----------------------
# Title
# -----------------------
st.title("📱 PhonePe Pulse — Dashboard & Insights")

# Logo + header
col_logo, col_title = st.columns([1, 10])
with col_logo:
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, width=60)
        except:
            pass
with col_title:
    st.markdown("### PhonePe Pulse — Visualization")

# -----------------------
# Locate SQLite DB
# -----------------------
DB_FILE_CANDIDATES = ["phonepe.db", "phonepe_pulse.db", "phone.db", "phonepe_pulse.sqlite"]
db_file = None
for f in DB_FILE_CANDIDATES:
    if os.path.exists(f):
        db_file = f
        break

if db_file is None:
    st.error("No phonepe sqlite DB found in the app folder. Place phonepe.db (or phonepe_pulse.db) here.")
    st.stop()

conn = sqlite3.connect(db_file, check_same_thread=False)

# -----------------------
# Helper: cached table loader
# -----------------------
@st.cache_data(ttl=600)
def load_table(sql):
    return pd.read_sql(sql, conn)

# Load tables
try:
    df_state = load_table("SELECT * FROM state_trans_user;")
    df_state_yearly = load_table("SELECT * FROM state_yearly_transactions;")
    df_district = load_table("SELECT * FROM district_transactions;")
    df_top = load_table("SELECT * FROM top_transactions;")
except Exception as e:
    st.error(f"Failed to load expected tables from DB: {e}")
    st.stop()

for df in [df_state, df_state_yearly, df_district, df_top]:
    if "states" in df.columns:
        df["states"] = df["states"].astype(str).str.strip().str.lower()

# -----------------------
# Load GeoJSON
# -----------------------
GEOFILE = "india_states.geojson"
if not os.path.exists(GEOFILE):
    st.error(f"GeoJSON file '{GEOFILE}' not found. Place it in project folder.")
    st.stop()

with open(GEOFILE, "r", encoding="utf-8") as f:
    india_geo = json.load(f)

# ---- GEOJSON normalization (unchanged code) ----
def sample_coord(coords):
    if coords is None:
        return None
    if isinstance(coords[0], (float, int)):
        return coords
    return sample_coord(coords[0])

def looks_like_lonlat(pair):
    if not pair or not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return True
    a, b = pair[0], pair[1]
    if abs(b) > 90 and abs(a) <= 90:
        return False
    if abs(a) > 180 or abs(b) > 180:
        return False
    return True

def swap_coords_recursively(coords):
    if isinstance(coords[0], (float, int)):
        return [coords[1], coords[0]]
    return [swap_coords_recursively(c) for c in coords]

def ensure_closed_ring(ring):
    if len(ring) < 4:
        return ring
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring

def normalize_geometry(geom):
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if coords is None:
        return geom
    try:
        sp = sample_coord(coords)
        if sp is not None and not looks_like_lonlat(sp):
            coords = swap_coords_recursively(coords)
    except:
        pass
    if gtype == "Polygon":
        new_coords = []
        for ring in coords:
            ring2 = [[float(pt[0]), float(pt[1])] for pt in ring]
            ring2 = ensure_closed_ring(ring2)
            new_coords.append(ring2)
        geom["coordinates"] = new_coords
    elif gtype == "MultiPolygon":
        new_mp = []
        for poly in coords:
            new_poly = []
            for ring in poly:
                ring2 = [[float(pt[0]), float(pt[1])] for pt in ring]
                ring2 = ensure_closed_ring(ring2)
                new_poly.append(ring2)
            new_mp.append(new_poly)
        geom["coordinates"] = new_mp
    if SHAPELY_AVAILABLE:
        try:
            shp = shape(geom)
            shp_valid = make_valid(shp)
            if isinstance(shp_valid, (Polygon, MultiPolygon)):
                shp_valid = shp_valid.simplify(0.0001, preserve_topology=True)
                geom2 = mapping(shp_valid)
                geom = {"type": geom2["type"], "coordinates": json.loads(json.dumps(geom2["coordinates"]))}
        except:
            pass
    return geom

fixed_features = []
for feat in india_geo.get("features", []):
    geom = feat.get("geometry", {})
    try:
        new_geom = normalize_geometry(geom)
        feat["geometry"] = new_geom
        props = feat.setdefault("properties", {})
        if "db_name" not in props:
            name = props.get("NAME_1") or props.get("st_nm") or props.get("NAME") or props.get("STATE_NAME") or ""
            dbn = str(name).strip().lower().replace("&", "and")
            props["db_name"] = dbn
        fixed_features.append(feat)
    except:
        pass
india_geo["features"] = fixed_features

def centroid_from_feature(feat):
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [])
    pts = []
    def collect(c):
        if not c:
            return
        if isinstance(c[0], (float, int)):
            pts.append((float(c[0]), float(c[1])))
        else:
            for it in c:
                collect(it)
    collect(coords)
    if not pts:
        return None, None
    lon = fsum([p[0] for p in pts]) / len(pts)
    lat = fsum([p[1] for p in pts]) / len(pts)
    return lon, lat

centroid_map = {}
for feat in india_geo.get("features", []):
    props = feat.get("properties", {})
    dbn = props.get("db_name") or (props.get("NAME_1") or "").strip().lower()
    lon, lat = centroid_from_feature(feat)
    if lon is not None:
        centroid_map[dbn] = {"lon": lon, "lat": lat}

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")
years = sorted(df_state["year"].unique())
year_selected = st.sidebar.selectbox("Year", years, index=len(years)-1)
quarters = sorted(df_state["quarter"].unique())
quarter_selected = st.sidebar.selectbox("Quarter", quarters, index=0)
types = sorted(df_state["trans_type"].unique())
type_selected = st.sidebar.selectbox("Transaction Type", types, index=0)

user_col = "user_counts" if "user_counts" in df_state.columns else ("registered_user_counts" if "registered_user_counts" in df_state.columns else None)

# -----------------------
# Prepare df_map
# -----------------------
df_filtered = df_state[
    (df_state["year"] == int(year_selected)) &
    (df_state["quarter"] == int(quarter_selected)) &
    (df_state["trans_type"] == type_selected)
].copy()

df_map = df_filtered.groupby("states", as_index=False).agg(
    total_amount=("amount", "sum"),
    total_transactions=("trans_counts", "sum")
)

df_map["states"] = df_map["states"].astype(str).str.strip().str.lower()
df_map["lon"] = df_map["states"].map(lambda s: centroid_map.get(s, {}).get("lon"))
df_map["lat"] = df_map["states"].map(lambda s: centroid_map.get(s, {}).get("lat"))
df_map["lon"] = df_map["lon"].fillna(78.9629)
df_map["lat"] = df_map["lat"].fillna(22.5937)

# -----------------------
# KPIs
# -----------------------
col1, col2, col3 = st.columns(3)
total_txn = int(df_map["total_transactions"].sum()) if not df_map.empty else 0
total_amt = float(df_map["total_amount"].sum()) if not df_map.empty else 0.0
avg_val = (total_amt / total_txn) if total_txn else 0
col1.metric("Total Transactions", f"{total_txn:,}")
col2.metric("Total Amount (₹)", f"{total_amt/1e9:.2f} B")
col3.metric("Avg Transaction Value (₹)", f"{avg_val:,.2f}")

# -----------------------
# 2D Choropleth
# -----------------------
st.subheader("India — Choropleth (by state)")
fig_ch = px.choropleth_mapbox(
    df_map,
    geojson=india_geo,
    locations="states",
    color="total_amount",
    featureidkey="properties.db_name",
    center={"lat": 22.0, "lon": 79.0},
    mapbox_style="carto-positron",
    zoom=3.6,
    opacity=0.7,
    color_continuous_scale="Viridis",
    hover_data={"total_amount": True, "total_transactions": True, "states": True}
)
fig_ch.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=520)
st.plotly_chart(fig_ch, use_container_width=True)

# -----------------------
# 3D Map (unchanged)
# -----------------------
st.subheader("3D Map — Extruded States + 3D Pillars (interactive)")

amount_lookup = dict(zip(df_map["states"], df_map["total_amount"]))
txn_lookup = dict(zip(df_map["states"], df_map["total_transactions"]))

min_amt = df_map["total_amount"].min() if not df_map.empty else 0.0
max_amt = df_map["total_amount"].max() if not df_map.empty else 1.0
rng = max(1.0, max_amt - min_amt)

def amt_to_color_list(amt):
    ratio = (amt - min_amt) / rng if rng > 0 else 0.0
    r = int(40 + 215 * ratio)
    g = int(220 - 180 * ratio)
    b = int(200 - 160 * ratio)
    return [max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))]

geo_with_amount = {"type": "FeatureCollection", "features": []}
for feat in india_geo.get("features", []):
    props = dict(feat.get("properties", {}))
    raw_name = props.get("NAME_1") or props.get("st_nm") or props.get("name") or props.get("db_name") or ""
    clean_name = str(raw_name).strip().lower()
    amt = float(amount_lookup.get(clean_name, 0.0))
    tx = int(txn_lookup.get(clean_name, 0))
    props["state_name"] = raw_name.strip().title()
    props["amount"] = round(amt, 2)
    props["transactions"] = int(tx)
    props["display_name"] = props["state_name"]
    props["fill_color"] = amt_to_color_list(amt)
    props["elevation"] = amt / (max_amt / 50000 + 1)
    geo_with_amount["features"].append({
        "type": feat.get("type"),
        "geometry": feat.get("geometry"),
        "properties": props
    })

col_rows = []
for _, row in df_map.iterrows():
    s = row["states"]
    lon = float(row["lon"])
    lat = float(row["lat"])
    amt = float(row["total_amount"])
    tx = int(row["total_transactions"])
    col_rows.append({
        "state": s.title(),
        "lon": lon,
        "lat": lat,
        "amount": amt,
        "transactions": tx,
        "color": amt_to_color_list(amt),
        "elev": float(amt) / (max_amt / 20000 + 1)
    })
df_cols = pd.DataFrame(col_rows)

geo_layer = pdk.Layer(
    "GeoJsonLayer",
    data=geo_with_amount,
    stroked=True,
    filled=True,
    extruded=True,
    wireframe=False,
    get_elevation="properties.elevation",
    elevation_scale=1.0,
    get_fill_color="properties.fill_color",
    pickable=True,
    auto_highlight=True,
)

col_layer = None
if not df_cols.empty:
    col_layer = pdk.Layer(
        "ColumnLayer",
        data=df_cols,
        get_position=["lon", "lat"],
        get_elevation="elev",
        elevation_scale=1.0,
        radius=30000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

view_state = pdk.ViewState(latitude=22.0, longitude=79.0, zoom=4.0, pitch=55, bearing=0)
layers = [geo_layer]
if col_layer:
    layers.append(col_layer)

tooltip_html = """
<b>{state_name}</b><br>
Amount: ₹{amount}<br>
Transactions: {transactions}
"""

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v9',
    tooltip={"html": tooltip_html, "style": {"backgroundColor": "steelblue", "color": "white"}}
)

st.pydeck_chart(deck)

# -----------------------
# Trends & Top lists (unchanged)
# -----------------------
st.subheader("Trends & Top lists")

df_trend = df_state[df_state["trans_type"] == type_selected].groupby("year", as_index=False).agg(
    total_amount=("amount", "sum"),
    total_transactions=("trans_counts", "sum")
).sort_values("year")
fig_trend = px.line(df_trend, x="year", y="total_amount", markers=True, title=f"Yearly Amount — {type_selected}")
st.plotly_chart(fig_trend, use_container_width=True)

df_top_states = df_map.sort_values("total_amount", ascending=False).head(10)
fig_top_states = px.bar(df_top_states, x="total_amount", y="states", orientation="h", color="total_amount", title="Top 10 States by Amount")
st.plotly_chart(fig_top_states, use_container_width=True)

st.subheader("Top districts (selected filters)")
df_d_filtered = df_district[(df_district["year"] == int(year_selected)) & (df_district["quarter"] == int(quarter_selected))].copy()
df_d_filtered["states"] = df_d_filtered["states"].astype(str).str.strip().str.lower()
top_d = df_d_filtered.groupby(["states", "district"], as_index=False).agg(total_amount=("amount", "sum"), total_transactions=("trans_counts", "sum")).sort_values("total_amount", ascending=False).head(10)
fig_top_d = px.bar(top_d, x="total_amount", y="district", orientation="h", color="total_amount", title="Top 10 Districts by Amount")
st.plotly_chart(fig_top_d, use_container_width=True)
st.dataframe(top_d.reset_index(drop=True))

# -----------------------
# SQL Insights
# -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("SQL Insights (Query Explorer)")

QUERY_CATALOG = [
    ("Q01 - Total amount per state (latest year, selected q)",
     "SELECT states, SUM(amount) AS total_amount FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_amount DESC;"),
    ("Q02 - Total transaction count per state (latest year, selected q)",
     "SELECT states, SUM(trans_counts) AS total_transactions FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_transactions DESC;"),
    ("Q03 - Total users per state (latest year, selected q)",
     "SELECT states, SUM(user_counts) AS total_users FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_users DESC;"),
    ("Q04 - Top 5 brands per state (latest year, selected q)",
     "SELECT states, brand, SUM(user_counts) AS total_users FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states, brand ORDER BY states, total_users DESC;"),
    ("Q05 - State yearly totals (all years)",
     "SELECT states, year, SUM(trans_counts) AS total_transactions, SUM(amount) AS total_amount FROM state_trans_user GROUP BY states, year ORDER BY states, year;"),
    ("Q06 - Yearly totals (India)",
     "SELECT year, SUM(trans_counts) AS total_transactions, SUM(amount) AS total_amount FROM state_trans_user GROUP BY year ORDER BY year;"),
    ("Q07 - Quarterly totals (all years)",
     "SELECT year, quarter, SUM(trans_counts) AS total_transactions, SUM(amount) AS total_amount FROM state_trans_user GROUP BY year, quarter ORDER BY year, quarter;"),
    ("Q08 - Top 10 states by YoY txn (latest year)",
     "WITH s AS (SELECT states, year, SUM(trans_counts) AS total_transactions FROM state_trans_user GROUP BY states, year) SELECT a.states, a.year, a.total_transactions, (a.total_transactions - b.total_transactions)*1.0 / NULLIF(b.total_transactions,0) AS yoy FROM s a LEFT JOIN s b ON a.states = b.states AND a.year = b.year + 1 WHERE a.year = {yr} ORDER BY yoy DESC LIMIT 10;"),
    ("Q09 - Top 10 states by YoY amount (latest year)",
     "WITH s AS (SELECT states, year, SUM(amount) AS total_amount FROM state_trans_user GROUP BY states, year) SELECT a.states, a.year, a.total_amount, (a.total_amount - b.total_amount)*1.0 / NULLIF(b.total_amount,0) AS yoy FROM s a LEFT JOIN s b ON a.states = b.states AND a.year = b.year + 1 WHERE a.year = {yr} ORDER BY yoy DESC LIMIT 10;"),
    ("Q10 - States with negative txn growth (latest year)",
     "WITH s AS (SELECT states, year, SUM(trans_counts) AS total_transactions FROM state_trans_user GROUP BY states, year) SELECT a.states, a.year, (a.total_transactions - b.total_transactions)*1.0 / NULLIF(b.total_transactions,0) AS yoy FROM s a LEFT JOIN s b ON a.states = b.states AND a.year = b.year + 1 WHERE a.year = {yr} AND ((a.total_transactions - b.total_transactions)*1.0 / NULLIF(b.total_transactions,0)) < 0 ORDER BY yoy ASC;"),
    ("Q11 - District totals per year",
     "SELECT states, district, year, SUM(trans_counts) AS total_transactions, SUM(amount) AS total_amount FROM district_transactions GROUP BY states, district, year ORDER BY year DESC, total_amount DESC LIMIT 100;"),
    ("Q12 - Top 10 districts by amount (latest filters)",
     "SELECT states, district, SUM(amount) AS total_amount FROM district_transactions WHERE year = {yr} AND quarter = {qr} GROUP BY states, district ORDER BY total_amount DESC LIMIT 10;"),
    ("Q13 - Top 10 districts by transactions (latest filters)",
     "SELECT states, district, SUM(trans_counts) AS total_transactions FROM district_transactions WHERE year = {yr} AND quarter = {qr} GROUP BY states, district ORDER BY total_transactions DESC LIMIT 10;"),
    ("Q14 - Top 10 states by registered users (latest filters)",
     "SELECT states, SUM(registered_user_counts) AS total_users FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_users DESC LIMIT 10;"),
    ("Q15 - Top 10 districts by registered users (latest filters)",
     "SELECT states, district, SUM(registered_user_counts) AS total_users FROM district_transactions WHERE year = {yr} AND quarter = {qr} GROUP BY states, district ORDER BY total_users DESC LIMIT 10;"),
]

query_names = [q[0] for q in QUERY_CATALOG]
selected_query_name = st.sidebar.selectbox("Choose a query to run (SQL Insights)", query_names)

sql_template = dict(QUERY_CATALOG)[selected_query_name]

top_state_example = df_map.sort_values("total_amount", ascending=False)["states"].head(1).iloc[0] if not df_map.empty else ""
params = {"yr": int(year_selected), "qr": int(quarter_selected), "state": top_state_example.replace("'", "''") if isinstance(top_state_example, str) else top_state_example}

try:
    sql_to_run = sql_template.format(**params)
except:
    sql_to_run = sql_template

st.sidebar.markdown("**SQL to run:**")
st.sidebar.code(sql_to_run)

st.header("SQL Insights — Results")
try:
    df_q = pd.read_sql(sql_to_run, conn)
    if USE_AGGRID:
        gb = GridOptionsBuilder.from_dataframe(df_q)
        gb.configure_default_column(filterable=True, sortable=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gridOptions = gb.build()
        AgGrid(df_q, gridOptions=gridOptions, enable_enterprise_modules=False)
    else:
        st.dataframe(df_q.head(200), use_container_width=True)

    numeric_cols = df_q.select_dtypes("number").columns.tolist()
    if numeric_cols:
        ycol = numeric_cols[0]
        if "states" in df_q.columns:
            label_col = "states" if "states" in df_q.columns else df_q.columns[0]
            fig_q = px.bar(df_q.sort_values(ycol, ascending=False).head(20), x=ycol, y=label_col, orientation="h")
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.plotly_chart(px.line(df_q, y=ycol, x=df_q.columns[0]) if len(df_q.columns) > 1 else px.bar(df_q, y=ycol), use_container_width=True)

except Exception as e:
    st.error(f"Query failed: {e}")

# ============================================================
# 📊 INSIGHTS & RECOMMENDATIONS (MERGED INTO MAIN APP)
# ============================================================

st.markdown("## 📊 Insights & Recommendations")
st.markdown("""
Welcome to the **Insights & Recommendations Dashboard**, derived from  
the **full EDA** and **50 SQL Queries** performed on the PhonePe Pulse dataset.

Below are the **business insights**, **patterns**, and **data-driven recommendations**.
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
- Bengaluru Urban  
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

# End of insights
st.markdown("---")
st.caption("Generated from EDA & 50 SQL Queries | PhonePe Pulse Data Analytics")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption(f"DB: {db_file} — Features loaded: {len(india_geo.get('features', []))} — Developer: Debashish Borah")

