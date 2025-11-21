import os
import json
import sqlite3
from math import fsum
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
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

# PIL for logo
try:
    from PIL import Image
except Exception:
    Image = None

# -----------------------
# Page config + Title
# -----------------------
# Try to set a page icon if images.jpeg exists
logo_path = "images.jpeg"
if os.path.exists(logo_path) and Image is not None:
    try:
        logo_img = Image.open(logo_path)
        st.set_page_config(page_title="PhonePe Pulse — Dashboard & Insights", page_icon=logo_img, layout="wide")
    except Exception:
        st.set_page_config(page_title="PhonePe Pulse — Dashboard & Insights", page_icon="📱", layout="wide")
else:
    st.set_page_config(page_title="PhonePe Pulse — Dashboard & Insights", page_icon="📱", layout="wide")

st.title("📱 PhonePe Pulse — Dashboard & Insights")

# show logo + title in header area
col_logo, col_title = st.columns([1, 10])
with col_logo:
    if os.path.exists(logo_path) and Image is not None:
        try:
            st.image(logo_path, width=60)
        except Exception:
            st.write("")  # ignore
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

# Connect
conn = sqlite3.connect(db_file, check_same_thread=False)

# -----------------------
# Helper: cached table loader
# -----------------------
@st.cache_data(ttl=600)
def load_table(sql):
    return pd.read_sql(sql, conn)

# Load expected tables; if missing, show error and stop
try:
    df_state = load_table("SELECT * FROM state_trans_user;")
    df_state_yearly = load_table("SELECT * FROM state_yearly_transactions;")
    df_district = load_table("SELECT * FROM district_transactions;")
    df_top = load_table("SELECT * FROM top_transactions;")
except Exception as e:
    st.error(f"Failed to load expected tables from DB: {e}")
    st.stop()

# normalize state names to lowercase trimmed for joins/lookups
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

# Normalize and fix geojson features:
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
    # If second value > 90 it's probably lat in second slot -> suspicious
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
    except Exception:
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
    else:
        geom["coordinates"] = coords
    # shapely validification (optional)
    if SHAPELY_AVAILABLE:
        try:
            shp = shape(geom)
            shp_valid = make_valid(shp)
            if isinstance(shp_valid, (Polygon, MultiPolygon)):
                shp_valid = shp_valid.simplify(0.0001, preserve_topology=True)
                geom2 = mapping(shp_valid)
                geom = {"type": geom2["type"], "coordinates": json.loads(json.dumps(geom2["coordinates"]))}
        except Exception:
            pass
    return geom

# Apply to features and ensure properties.db_name exists
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
    except Exception:
        # skip broken
        pass
india_geo["features"] = fixed_features

# Compute centroid (lon,lat) for each feature (simple average)
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

# user counts column (compat)
user_col = "user_counts" if "user_counts" in df_state.columns else ("registered_user_counts" if "registered_user_counts" in df_state.columns else None)

# -----------------------
# Prepare df_map (aggregated)
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
# attach centroids (fallback to India center if missing)
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
# 2D Choropleth (Plotly)
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
# 3D Map: GeoJsonLayer (extruded) + ColumnLayer (pillars)
# PRECOMPUTE colors and elevations in feature properties to avoid JSON expressions
# -----------------------
st.subheader("3D Map — Extruded States + 3D Pillars (interactive)")

# lookups
amount_lookup = dict(zip(df_map["states"], df_map["total_amount"]))
txn_lookup = dict(zip(df_map["states"], df_map["total_transactions"]))

min_amt = df_map["total_amount"].min() if not df_map.empty else 0.0
max_amt = df_map["total_amount"].max() if not df_map.empty else 1.0
rng = max(1.0, max_amt - min_amt)

def amt_to_color_list(amt):
    # returns [r,g,b] integers
    ratio = (amt - min_amt) / rng if rng > 0 else 0.0
    r = int(40 + 215 * ratio)
    g = int(220 - 180 * ratio)
    b = int(200 - 160 * ratio)
    return [max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))]

# Create a geojson copy with precomputed props
geo_with_amount = {"type": "FeatureCollection", "features": []}
for feat in india_geo.get("features", []):
    props = dict(feat.get("properties", {}))
    # choose display name from NAME_1 / st_nm / db_name
    raw_name = props.get("NAME_1") or props.get("st_nm") or props.get("name") or props.get("db_name") or ""
    clean_name = str(raw_name).strip().lower()
    amt = float(amount_lookup.get(clean_name, 0.0))
    tx = int(txn_lookup.get(clean_name, 0))
    # Tooltip-safe properties
    props["state_name"] = raw_name.strip().title()     # <-- State name for tooltip
    props["amount"] = round(amt, 2)                    # <-- Amount for tooltip
    props["transactions"] = int(tx)                    # <-- Transactions for tooltip

    # Map styling properties
    props["display_name"] = props["state_name"]        # used by map fill
    props["fill_color"] = amt_to_color_list(amt)
    props["elevation"] = amt / (max_amt / 50000 + 1)

    geo_with_amount["features"].append({
        "type": feat.get("type"),
        "geometry": feat.get("geometry"),
        "properties": props
    })

# Column layer data (centroids)
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

# Build layers
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

# Tooltip: use properties for GeoJsonLayer and field names for ColumnLayer
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
# Trends & Top lists (same visuals you had)
# -----------------------
st.subheader("Trends & Top lists")

# Trend line for selected trans_type
df_trend = df_state[df_state["trans_type"] == type_selected].groupby("year", as_index=False).agg(
    total_amount=("amount", "sum"),
    total_transactions=("trans_counts", "sum")
).sort_values("year")
fig_trend = px.line(df_trend, x="year", y="total_amount", markers=True, title=f"Yearly Amount — {type_selected}")
st.plotly_chart(fig_trend, use_container_width=True)

# Top 10 states by amount
df_top_states = df_map.sort_values("total_amount", ascending=False).head(10)
fig_top_states = px.bar(df_top_states, x="total_amount", y="states", orientation="h", color="total_amount", title="Top 10 States by Amount")
st.plotly_chart(fig_top_states, use_container_width=True)

# Top 10 districts by amount (latest year/quarter)
st.subheader("Top districts (selected filters)")
df_d_filtered = df_district[(df_district["year"] == int(year_selected)) & (df_district["quarter"] == int(quarter_selected))].copy()
df_d_filtered["states"] = df_d_filtered["states"].astype(str).str.strip().str.lower()
top_d = df_d_filtered.groupby(["states", "district"], as_index=False).agg(total_amount=("amount", "sum"), total_transactions=("trans_counts", "sum")).sort_values("total_amount", ascending=False).head(10)
fig_top_d = px.bar(top_d, x="total_amount", y="district", orientation="h", color="total_amount", title="Top 10 Districts by Amount")
st.plotly_chart(fig_top_d, use_container_width=True)
st.dataframe(top_d.reset_index(drop=True))

# -----------------------
# SQL Insights (Query Explorer)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("SQL Insights (Query Explorer)")

# Query catalog (same as before) — templates use {yr} {qr} {state}
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
    # ... you can append the rest of your catalog
]

query_names = [q[0] for q in QUERY_CATALOG]
selected_query_name = st.sidebar.selectbox("Choose a query to run (SQL Insights)", query_names)

sql_template = dict(QUERY_CATALOG)[selected_query_name]
# default example state for templates
top_state_example = df_map.sort_values("total_amount", ascending=False)["states"].head(1).iloc[0] if not df_map.empty else ""
params = {"yr": int(year_selected), "qr": int(quarter_selected), "state": top_state_example.replace("'", "''") if isinstance(top_state_example, str) else top_state_example}
try:
    sql_to_run = sql_template.format(**params)
except Exception:
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
            # attempt to find best label column
            label_col = "states" if "states" in df_q.columns else df_q.columns[0]
            fig_q = px.bar(df_q.sort_values(ycol, ascending=False).head(20), x=ycol, y=label_col, orientation="h")
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.plotly_chart(px.line(df_q, y=ycol, x=df_q.columns[0]) if len(df_q.columns) > 1 else px.bar(df_q, y=ycol), use_container_width=True)
except Exception as e:
    st.error(f"Query failed: {e}")

# -----------------------
# Footer / notes
# -----------------------
st.markdown("---")

st.caption(f"DB: {db_file} — Features loaded: {len(india_geo.get('features', []))} — Developer: Debashish Borah")
