import os
import json
import sqlite3
from math import fsum
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import time

st.write("APP STARTED — DEBUG CHECKPOINT 1")

try:
    import pandas as pd
    st.write("Pandas imported — DEBUG CHECKPOINT 2")
except Exception as e:
    st.error(f"Pandas import FAIL: {e}")

st.write("Before loading geojson — DEBUG CHECKPOINT 3")

# ------------------------------
# Load & display PhonePe Logo
# ------------------------------
from PIL import Image

logo_path = "images.jpeg"

try:
    phonepe_logo = Image.open(logo_path)
    st.set_page_config(
        page_title="PhonePe Pulse Dashboard",
        page_icon=phonepe_logo
    )
except:
    # Fallback (Streamlit Cloud may require static icon)
    st.set_page_config(
        page_title="PhonePe Pulse Dashboard",
        page_icon="📱"
    )

# pydeck
try:
    import pydeck as pdk
except Exception:
    st.error("pydeck is required. Install with: pip install pydeck")
    raise

# aggrid optional
USE_AGGRID = False
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    USE_AGGRID = True
except Exception:
    USE_AGGRID = False

# shapely optional (better fixes)
SHAPELY_AVAILABLE = True
try:
    from shapely.geometry import shape, mapping, Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid
except Exception:
    SHAPELY_AVAILABLE = False

st.set_page_config(page_title="PhonePe Pulse — Final", layout="wide")
st.title(" PhonePe Pulse — DASHBOARD (3D)")
col1, col2 = st.columns([1, 8])
with col1:
    st.image("images.jpeg", width=60)
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>PhonePe Pulse - Dashboard</h1>", unsafe_allow_html=True)


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
# Load tables (cached)
# -----------------------
@st.cache_data(ttl=600)
def load_table(sql):
    return pd.read_sql(sql, conn)

try:
    df_state = load_table("SELECT * FROM state_trans_user;")
    df_state_yearly = load_table("SELECT * FROM state_yearly_transactions;")
    df_district = load_table("SELECT * FROM district_transactions;")
    df_top = load_table("SELECT * FROM top_transactions;")
except Exception as e:
    st.error(f"Failed to load expected tables from DB: {e}")
    st.stop()

# normalize state names
for df in [df_state, df_state_yearly, df_district, df_top]:
    if "states" in df.columns:
        df["states"] = df["states"].astype(str).str.strip().str.lower()

# -----------------------
# Load GeoJSON
# -----------------------
st.write("DEBUG CHECKPOINT 3 — Preparing to load GeoJSON")

GEOFILE = "india_states.geojson"

if not os.path.exists(GEOFILE):
    st.error(f"ERROR: {GEOFILE} not found in project folder.")
    st.stop()

st.write("DEBUG CHECKPOINT 4 — GeoJSON file found, trying to open it")

try:
    with open(GEOFILE, "r", encoding="utf-8") as f:
        india_geo = json.load(f)

    st.write("DEBUG CHECKPOINT 5 — GeoJSON loaded successfully")
    st.write("DEBUG — First feature properties:", india_geo["features"][0]["properties"])
    st.info(f"Loaded geojson features: {len(india_geo.get('features', []))}. Shapely available: {SHAPELY_AVAILABLE}")

except Exception as e:
    st.error(f"GeoJSON LOAD FAILED — {e}")
    st.stop()

# -----------------------
# Geometry repair utilities
# -----------------------

def sample_coord(coords):
    # return a sample coordinate deep in structure
    if coords is None:
        return None
    if isinstance(coords[0], (float, int)):
        return coords
    return sample_coord(coords[0])

def looks_like_lonlat(pair):
    # True if pair appears as [lon, lat]
    if not pair or not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return True
    a, b = pair[0], pair[1]
    # If second value > 90 it's probably lon placed in second slot -> wrong
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
    """
    Ensure geometry coordinates are lon,lat and polygon rings closed.
    If shapely available, we run make_valid + simplify to remove self-intersections.
    """
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if coords is None:
        return geom
    # Sample to check order
    try:
        sp = sample_coord(coords)
        if sp is not None and not looks_like_lonlat(sp):
            coords = swap_coords_recursively(coords)
    except Exception:
        pass

    # close rings for Polygon/MultiPolygon
    if gtype == "Polygon":
        new_coords = []
        for ring in coords:
            ring = [[float(pt[0]), float(pt[1])] for pt in ring]
            ring = ensure_closed_ring(ring)
            new_coords.append(ring)
        geom["coordinates"] = new_coords
    elif gtype == "MultiPolygon":
        new_mp = []
        for poly in coords:
            new_poly = []
            for ring in poly:
                ring = [[float(pt[0]), float(pt[1])] for pt in ring]
                ring = ensure_closed_ring(ring)
                new_poly.append(ring)
            new_mp.append(new_poly)
        geom["coordinates"] = new_mp
    else:
        geom["coordinates"] = coords
    # if shapely available, attempt make_valid and return simplified mapping
    if SHAPELY_AVAILABLE:
        try:
            shp = shape(geom)
            shp_valid = make_valid(shp)
            # unify small components
            if isinstance(shp_valid, (Polygon, MultiPolygon)):
                # optionally simplify slightly for performance
                shp_valid = shp_valid.simplify(0.0001, preserve_topology=True)
                geom2 = mapping(shp_valid)
                # mapping returns coords as floats; ensure lists
                geom = {"type": geom2["type"], "coordinates": json.loads(json.dumps(geom2["coordinates"]))}
        except Exception:
            # if shapely fails, keep previous geom
            pass
    return geom

# Apply normalization to all features (idempotent)
fixed_features = []
for feat in india_geo.get("features", []):
    geom = feat.get("geometry", {})
    try:
        new_geom = normalize_geometry(geom)
        feat["geometry"] = new_geom
        # ensure properties.db_name exists
        props = feat.setdefault("properties", {})
        if "db_name" not in props:
            name = props.get("NAME_1") or props.get("st_nm") or props.get("NAME") or props.get("STATE_NAME") or ""
            # normalize to lower-case matching our DB
            dbn = str(name).strip().lower()
            dbn = dbn.replace("&", "and")
            props["db_name"] = dbn
        fixed_features.append(feat)
    except Exception:
        # skip broken feature
        pass

india_geo["features"] = fixed_features
st.success(f"GeoJSON normalized. Features remaining: {len(india_geo.get('features', []))}")

# -----------------------
# Compute centroids (lon, lat) for each feature
# -----------------------
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
# Build df_map for selected filters (map data)
# -----------------------
# Sidebar filters
st.sidebar.header("Filters")
years = sorted(df_state["year"].unique())
year_selected = st.sidebar.selectbox("Year", years, index=len(years)-1)
quarters = sorted(df_state["quarter"].unique())
quarter_selected = st.sidebar.selectbox("Quarter", quarters, index=0)
types = sorted(df_state["trans_type"].unique())
type_selected = st.sidebar.selectbox("Transaction Type", types, index=0)

# Filter df_state
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

# Safe defaults if no centroid found: use approximate center of India
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
# 2D Choropleth (plotly)
# -----------------------
st.subheader("India — Choropleth (by state)")
# Ensure properties.db_name exists for all features (done above)
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
# 3D Map (pydeck) — Extruded states + Pillars
# -----------------------
st.subheader("3D Map — Extruded States + 3D Pillars (Interactive)")

# Build lookups
amount_lookup = dict(zip(df_map["states"], df_map["total_amount"]))
txn_lookup = dict(zip(df_map["states"], df_map["total_transactions"]))

min_amt = df_map["total_amount"].min() if not df_map.empty else 0
max_amt = df_map["total_amount"].max() if not df_map.empty else 1
rng = max(1, max_amt - min_amt)

def amt_to_color_list(amt):
    ratio = (amt - min_amt) / rng if rng > 0 else 0
    r = int(40 + 215 * ratio)
    g = int(220 - 180 * ratio)
    b = int(200 - 160 * ratio)
    return [max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))]

# -----------------------
# FIX: Use correct GeoJSON property for DISPLAY NAME
# -----------------------
geo_with_amount = {"type": "FeatureCollection", "features": []}

for feat in india_geo.get("features", []):
    props = dict(feat.get("properties", {}))

    # Pick the best available state name from multiple keys
    raw_name = (
        props.get("NAME_1")
        or props.get("ST_NM")
        or props.get("state_name")
        or props.get("name")
        or ""
    )

    clean_name = raw_name.strip().lower()

    amt = float(amount_lookup.get(clean_name, 0))
    tx = int(txn_lookup.get(clean_name, 0))

    props["amount"] = amt
    props["transactions"] = tx

    # FIX: display real name, not db_name
    props["display_name"] = raw_name.strip().title()

    props["fill_color"] = amt_to_color_list(amt)
    props["elevation"] = amt / (max_amt / 50000 + 1)

    geo_with_amount["features"].append({
        "type": "Feature",
        "geometry": feat.get("geometry"),
        "properties": props
    })

# -----------------------
# Pillar Layer (State centroids)
# -----------------------
col_rows = []
for _, row in df_map.iterrows():
    col_rows.append({
        "state": row["states"].title(),
        "lon": float(row["lon"]),
        "lat": float(row["lat"]),
        "amount": float(row["total_amount"]),
        "transactions": int(row["total_transactions"]),
        "color": amt_to_color_list(float(row["total_amount"])),
        "elev": float(row["total_amount"]) / (max_amt / 30000 + 1)
    })

df_cols = pd.DataFrame(col_rows)

# -----------------------
# Deck.gl Layers
# -----------------------
poly_layer = pdk.Layer(
    "GeoJsonLayer",
    data=geo_with_amount,
    stroked=True,
    filled=True,
    extruded=True,
    wireframe=False,
    get_elevation="properties.elevation",
    get_fill_color="properties.fill_color",
    pickable=True,
    auto_highlight=True,
)

col_layer = pdk.Layer(
    "ColumnLayer",
    data=df_cols,
    get_position=["lon", "lat"],
    get_elevation="elev",
    radius=30000,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=22.0,
    longitude=79.0,
    zoom=4.0,
    pitch=55,
    bearing=0,
)

# -----------------------
# FIX: Tooltip uses correct names
# -----------------------
tooltip_html = """
<b>{display_name}</b><br>
Amount: ₹{amount}<br>
Transactions: {transactions}
"""

deck = pdk.Deck(
    layers=[poly_layer, col_layer],
    initial_view_state=view_state,
    tooltip={"html": tooltip_html},
    map_style='mapbox://styles/mapbox/light-v9'
)

st.pydeck_chart(deck)


# -----------------------
# Additional visuals (trends, top lists)
# -----------------------
st.subheader("Trends & Top lists")
df_trend = df_state[df_state["trans_type"] == type_selected].groupby("year", as_index=False).agg(total_amount=("amount", "sum"), total_transactions=("trans_counts", "sum"))
fig_trend = px.line(df_trend, x="year", y="total_amount", markers=True, title=f"Yearly amount — {type_selected}")
st.plotly_chart(fig_trend, use_container_width=True)

df_top_states = df_map.sort_values("total_amount", ascending=False).head(10)
fig_top = px.bar(df_top_states, x="total_amount", y="states", orientation="h", color="total_amount", title="Top 10 States by Amount")
st.plotly_chart(fig_top, use_container_width=True)

st.subheader("Top districts (selected filters)")
df_d_filtered = df_district[
    (df_district["year"] == year_selected) &
    (df_district["quarter"] == quarter_selected)
].copy()
df_d_filtered["states"] = df_d_filtered["states"].astype(str).str.strip().str.lower()
top_d = df_d_filtered.groupby(["states", "district"], as_index=False).agg(total_amount=("amount", "sum"), total_transactions=("trans_counts", "sum")).sort_values("total_amount", ascending=False).head(10)
fig_d = px.bar(top_d, x="total_amount", y="district", orientation="h", color="total_amount", title="Top 10 Districts (Amount)")
st.plotly_chart(fig_d, use_container_width=True)
st.dataframe(top_d.reset_index(drop=True))

# -----------------------
# SQL Insights page (Query Explorer)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("SQL Insights (Query Explorer)")

# NOTE: Keep the same query catalog you used earlier (shortened here for brevity)
QUERY_CATALOG = [
    ("Q01 - Total amount per state (latest year, selected q)",
     "SELECT states, SUM(amount) AS total_amount FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_amount DESC;"),
    ("Q02 - Total transaction count per state (latest year, selected q)",
     "SELECT states, SUM(trans_counts) AS total_transactions FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_transactions DESC;"),
    ("Q03 - Total users per state (latest year, selected q)",
     "SELECT states, SUM(user_counts) AS total_users FROM state_trans_user WHERE year = {yr} AND quarter = {qr} GROUP BY states ORDER BY total_users DESC;"),
    # ... add more queries or reuse your catalogue
]

query_names = [q[0] for q in QUERY_CATALOG]
selected_query_name = st.sidebar.selectbox("Choose a query to run (SQL Insights)", query_names)
sql_template = dict(QUERY_CATALOG)[selected_query_name]
params = {"yr": year_selected, "qr": quarter_selected, "state": ""}
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
            fig_q = px.bar(df_q.sort_values(ycol, ascending=False).head(20), x=ycol, y=df_q.columns[0] if df_q.columns[0] != ycol else df_q.columns[1], orientation="h")
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.plotly_chart(px.line(df_q, y=ycol, x=df_q.columns[0]) if len(df_q.columns) > 1 else px.bar(df_q, y=ycol), use_container_width=True)
except Exception as e:
    st.error(f"Query failed: {e}")

st.markdown("---")
st.caption(f"DB: {db_file} — Geo features: {len(india_geo.get('features', []))}")
st.markdown("---")
st.caption(f"DB: {db_file} — Tables loaded: state_trans_user, state_yearly_transactions, district_transactions, top_transactions. All is done by DEBASHISH BORAH")
