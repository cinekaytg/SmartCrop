import sqlite3
import itertools
from pathlib import Path
from datetime import date

import requests

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="SmartCrop | Field Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "smartcrop.db"
DEFAULT_LAT, DEFAULT_LON = 36.8969, 30.7133

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_power_monthly(latitude, longitude, year=2025):
    """Fetch daily NASA POWER data for one selected global point and aggregate monthly."""
    endpoint = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": round(float(longitude), 4),
        "latitude": round(float(latitude), 4),
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
        "time-standard": "LST",
    }
    response = requests.get(endpoint, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if payload.get("messages"):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            raise ValueError("; ".join(str(item) for item in messages))
    parameter_data = payload.get("properties", {}).get("parameter", {})
    if not parameter_data:
        raise ValueError("NASA POWER returned no parameter data for this location. Try another point.")

    frames = []
    for parameter, values in parameter_data.items():
        series = pd.Series(values, name=parameter)
        series.index = pd.to_datetime(series.index, format="%Y%m%d", errors="coerce")
        frames.append(series)

    daily = pd.concat(frames, axis=1).sort_index()
    daily = daily.apply(pd.to_numeric, errors="coerce").replace(-999, float("nan"))
    daily = daily.dropna(how="all")
    if daily.empty:
        raise ValueError("NASA POWER returned no valid daily observations for this location.")

    monthly = daily.resample("MS").agg({
        "T2M": "mean",
        "PRECTOTCORR": "sum",
        "ALLSKY_SFC_SW_DWN": "mean",
    }).reset_index().rename(columns={
        "index": "month_date",
        "T2M": "avg_temperature",
        "PRECTOTCORR": "total_rainfall",
        "ALLSKY_SFC_SW_DWN": "avg_solar_radiation",
    })
    if "month_date" not in monthly.columns:
        monthly = monthly.rename(columns={monthly.columns[0]: "month_date"})
    monthly["year_month"] = monthly["month_date"].dt.strftime("%Y-%m")
    return monthly[["year_month", "avg_temperature", "total_rainfall", "avg_solar_radiation"]]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f7f3;color:#20352b;}
[data-testid="stSidebar"] {background:#10382d;}
[data-testid="stSidebar"] * {color:#f4f8f4 !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {font-weight:500;}
.hero {background:linear-gradient(120deg,#10382d,#28734f);padding:30px 34px;border-radius:18px;color:white;margin-bottom:22px;}
.hero h1 {color:white;font-family:Manrope,sans-serif;font-size:2.15rem;margin:5px 0 8px;font-weight:800;}
.hero p {color:#e0eee5;margin:0;font-size:1rem;}
.eyebrow {letter-spacing:2px;font-size:11px;color:#c1dfcb;font-weight:700;}
.section {font-family:Manrope,sans-serif;color:#164332;font-size:1.18rem;font-weight:800;margin:14px 0 10px;}
div[data-testid="stMetric"] {background:#ffffff;border:1px solid #e0e9e1;padding:14px 16px;border-radius:14px;box-shadow:0 2px 8px rgba(20,60,40,.035);}
div[data-testid="stMetric"] label {color:#52665b !important;font-weight:600 !important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#173f2e !important;}
div[data-testid="stDataFrame"] {border:1px solid #e1e8e1;border-radius:12px;overflow:hidden;}
div[data-testid="stAlert"] {border-radius:12px;}
.small-note {color:#68796f;font-size:.88rem;}
.rotation-card {background:#fff;border:1px solid #e0e8e0;border-radius:14px;padding:16px 18px;margin:9px 0;}
.rotation-card.top {border:2px solid #2d8057;background:#f7fbf7;}
.rotation-sequence {font-size:1.08rem;font-weight:700;color:#183d2d;}
.rotation-groups {font-size:.86rem;color:#68796f;margin-top:7px;}
.footer {color:#718177;font-size:.82rem;text-align:center;padding:14px 0 4px;}
</style>
""", unsafe_allow_html=True)


def get_tables():
    with sqlite3.connect(DB_PATH) as con:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]


@st.cache_data(ttl=60)
def read_table(table):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def data(table):
    if table not in TABLES:
        return pd.DataFrame()
    try:
        return read_table(table)
    except Exception as exc:
        st.warning(f"Could not read {table}: {exc}")
        return pd.DataFrame()


def col(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">SMARTCROP · FIELD INTELLIGENCE</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def fmt(value, digits=2):
    try:
        if pd.isna(value):
            return "Not available"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "Not available"


def pretty_crop(value):
    names = {
        "barley": "Barley", "wheat": "Wheat", "maize": "Maize",
        "cotton": "Cotton", "sunflower": "Sunflower",
        "chickpea": "Chickpea", "tomato": "Tomato", "olive": "Olive",
        "cereal": "Cereal", "oilseed": "Oilseed",
        "vegetable": "Vegetable", "fiber": "Fiber",
        "legume": "Legume", "perennial": "Perennial",
    }
    if value is None or pd.isna(value):
        return "Not available"
    text = str(value)
    return names.get(text.lower(), text.replace("_", " ").title())


def range_fit(value, opt_min, opt_max, abs_min, abs_max):
    """Transparent 0-100 suitability fit from ECOCROP optimum and absolute ranges."""
    vals = [value, opt_min, opt_max, abs_min, abs_max]
    if any(v is None or pd.isna(v) for v in vals):
        return None
    value, opt_min, opt_max, abs_min, abs_max = map(float, vals)
    if opt_min <= value <= opt_max:
        return 100.0
    if value < opt_min:
        if value < abs_min or opt_min == abs_min:
            return 0.0
        return max(0.0, 100.0 * (value - abs_min) / (opt_min - abs_min))
    if value > opt_max:
        if value > abs_max or abs_max == opt_max:
            return 0.0
        return max(0.0, 100.0 * (abs_max - value) / (abs_max - opt_max))
    return 0.0


def build_location_suitability(climate_df, ecological, catalog, soil_ph=None):
    """Build a location-specific preliminary screening from available data only."""
    if climate_df.empty or ecological.empty:
        return pd.DataFrame()
    temp_col = col(climate_df, ["avg_temperature"])
    rain_col = col(climate_df, ["total_rainfall"])
    if not temp_col or not rain_col:
        return pd.DataFrame()
    annual_temp = pd.to_numeric(climate_df[temp_col], errors="coerce").mean()
    annual_rain = pd.to_numeric(climate_df[rain_col], errors="coerce").sum()
    rows = []
    for _, crop in ecological.iterrows():
        temp_score = range_fit(
            annual_temp, crop.get("temp_opt_min_c"), crop.get("temp_opt_max_c"),
            crop.get("temp_abs_min_c"), crop.get("temp_abs_max_c")
        )
        rain_score = range_fit(
            annual_rain, crop.get("annual_rain_opt_min_mm"),
            crop.get("annual_rain_opt_max_mm"),
            crop.get("annual_rain_abs_min_mm"), crop.get("annual_rain_abs_max_mm")
        )
        ph_score = None
        if soil_ph is not None and not pd.isna(soil_ph):
            ph_score = range_fit(
                float(soil_ph), crop.get("ph_opt_min"), crop.get("ph_opt_max"),
                crop.get("ph_abs_min"), crop.get("ph_abs_max")
            )
        components = [v for v in [temp_score, rain_score, ph_score] if v is not None]
        score = sum(components) / len(components) if components else None
        cat = catalog[catalog["crop_id"] == crop["crop_id"]]
        crop_name = cat.iloc[0]["crop_name_en"] if not cat.empty else str(crop["crop_id"]).title()
        crop_group = cat.iloc[0]["crop_group"] if not cat.empty else "Unknown"
        rows.append({
            "crop_id": crop["crop_id"],
            "Crop": crop_name,
            "Crop group": crop_group,
            "Temperature fit": temp_score,
            "Annual rainfall fit": rain_score,
            "Soil pH fit": ph_score,
            "Preliminary score": score,
            "Data components": len(components),
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Preliminary score", ascending=False, na_position="last")


def build_location_rotations(suitability):
    """Generate 3-year non-repeating sequences with no adjacent same crop group."""
    if suitability.empty:
        return pd.DataFrame()
    valid = suitability.dropna(subset=["Preliminary score"]).to_dict("records")
    candidates = []
    for a, b, c in itertools.permutations(valid, 3):
        if a["crop_id"] == b["crop_id"] or b["crop_id"] == c["crop_id"] or a["crop_id"] == c["crop_id"]:
            continue
        if a["Crop group"] == b["Crop group"] or b["Crop group"] == c["Crop group"]:
            continue
        scores = [a["Preliminary score"], b["Preliminary score"], c["Preliminary score"]]
        candidates.append({
            "Year 1": a["Crop"], "Year 2": b["Crop"], "Year 3": c["Crop"],
            "Crop group sequence": f'{a["Crop group"]} → {b["Crop group"]} → {c["Crop group"]}',
            "Mean screening score": sum(scores) / 3,
        })
    return pd.DataFrame(candidates).sort_values(
        "Mean screening score", ascending=False
    ).head(15) if candidates else pd.DataFrame()


if not DB_PATH.exists():
    st.error("Database file not found. Place app.py and smartcrop.db in the same folder.")
    st.code(str(DB_PATH))
    st.stop()

try:
    TABLES = get_tables()
except Exception as exc:
    st.error(f"Could not open the database: {exc}")
    st.stop()

# Global point selection. Climate data are fetched for the selected point;
# existing soil/crop/rotation tables remain the Antalya pilot dataset for now.
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = DEFAULT_LAT
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = DEFAULT_LON
if "selected_climate" not in st.session_state:
    st.session_state.selected_climate = pd.DataFrame()
if "climate_location" not in st.session_state:
    st.session_state.climate_location = None
if "location_suitability" not in st.session_state:
    st.session_state.location_suitability = pd.DataFrame()
if "location_rotations" not in st.session_state:
    st.session_state.location_rotations = pd.DataFrame()
if "location_soil_ph" not in st.session_state:
    st.session_state.location_soil_ph = None

with st.sidebar:
    st.markdown("# 🌱 SmartCrop")
    st.caption("NASA Space Apps Challenge")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Climate Analysis",
            "Soil Analysis",
            "Crop Suitability",
            "Crop Rotation",
            "Data Sources",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("SELECTED GLOBAL LOCATION")
    st.markdown(f"**{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}**")
    st.caption("NASA POWER climate data can be requested for the selected point.")

climate = data("climate_monthly")
soil = data("soil_site_summary")
profile = data("soil_profile")
suit = data("crop_suitability_assessment")
rotation = data("crop_rotation_candidates")
root = data("crop_root_zone_water")
ecological = data("crop_ecological_requirements")
catalog = data("crop_catalog")

if page == "Overview":
    hero(
        "Smarter decisions for healthier fields.",
        "Explore satellite-derived climate and soil information, compare crop requirements, and review transparent rotation scenarios.",
    )
    overview_location_current = (
        not st.session_state.selected_climate.empty
        and st.session_state.climate_location == (
            st.session_state.selected_lat, st.session_state.selected_lon
        )
    )
    if overview_location_current and not st.session_state.location_suitability.empty:
        overview_scores = st.session_state.location_suitability.copy()
        overview_name_col = "Crop"
        overview_score_col = "Preliminary score"
        overview_rotation_count = len(st.session_state.location_rotations)
        overview_climate_count = len(st.session_state.selected_climate)
    else:
        overview_scores = suit.copy()
        overview_name_col = col(overview_scores, ["crop_name_en", "crop_name", "name_en", "crop_id"])
        overview_score_col = col(overview_scores, ["preliminary_screening_score", "preliminary_score"])
        overview_rotation_count = len(rotation)
        overview_climate_count = len(climate)

    valid_scores = (
        pd.to_numeric(overview_scores[overview_score_col], errors="coerce").notna().sum()
        if overview_score_col and not overview_scores.empty else 0
    )
    a, b, c, d = st.columns(4)
    a.metric("Selected point", "Global")
    b.metric("Climate records", f"{overview_climate_count} months" if overview_climate_count else "Unavailable")
    c.metric("Scored crops", str(valid_scores))
    d.metric("Rotation candidates", str(overview_rotation_count))

    left, right = st.columns([1.12, 1])
    with left:
        section("Select any location worldwide")
        st.caption("Click anywhere on the map to choose a point. Then load its NASA POWER climate data.")
        map_view = folium.Map(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            zoom_start=2 if st.session_state.climate_location is None else 5,
            tiles="OpenStreetMap",
            min_zoom=1,
            max_zoom=18,
        )
        folium.Marker(
            [st.session_state.selected_lat, st.session_state.selected_lon],
            tooltip="Selected SmartCrop location",
            popup=f"Selected point: {st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}",
            icon=folium.Icon(color="green"),
        ).add_to(map_view)
        map_result = st_folium(
            map_view,
            height=390,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="global_location_map",
        )
        clicked = map_result.get("last_clicked") if map_result else None
        if clicked:
            new_lat = float(clicked["lat"])
            new_lon = float(clicked["lng"])
            if (abs(new_lat - st.session_state.selected_lat) > 0.0001 or
                    abs(new_lon - st.session_state.selected_lon) > 0.0001):
                st.session_state.selected_lat = new_lat
                st.session_state.selected_lon = new_lon
                st.session_state.selected_climate = pd.DataFrame()
                st.session_state.climate_location = None
                st.session_state.location_suitability = pd.DataFrame()
                st.session_state.location_rotations = pd.DataFrame()
                st.session_state.location_soil_ph = None
                st.rerun()

        st.write(
            f"**Selected coordinates:** {st.session_state.selected_lat:.4f}, "
            f"{st.session_state.selected_lon:.4f}"
        )
        if st.button("Load climate data for this location", type="primary", use_container_width=True):
            try:
                with st.spinner("Requesting NASA POWER data for the selected coordinates..."):
                    result = fetch_nasa_power_monthly(
                        st.session_state.selected_lat, st.session_state.selected_lon, 2025
                    )
                st.session_state.selected_climate = result
                st.session_state.climate_location = (
                    st.session_state.selected_lat, st.session_state.selected_lon
                )
                # SoilGrids point-query API is currently paused. Do not reuse Antalya pH
                # for another point. Climate-only screening is explicitly labelled.
                st.session_state.location_suitability = build_location_suitability(
                    result, ecological, catalog, soil_ph=None
                )
                st.session_state.location_rotations = build_location_rotations(
                    st.session_state.location_suitability
                )
                st.success("NASA POWER climate data loaded. Crop screening and rotation scenarios were recalculated using this location's climate and available ECOCROP requirements.")
            except Exception as exc:
                st.error(f"NASA POWER data could not be loaded: {exc}")
        st.caption(
            "Climate data and preliminary crop screening update for the selected point. "
            "Soil analysis remains limited to the Antalya pilot dataset."
        )
    with right:
        section("Preliminary crop screening")
        if overview_location_current and overview_score_col and overview_name_col:
            st.caption(
                f"Live climate-based screening · "
                f"{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}"
            )
        else:
            st.caption("Original Antalya pilot assessment · load climate data for a selected point to refresh")
        if overview_score_col and overview_name_col and not overview_scores.empty:
            view = overview_scores[[overview_name_col, overview_score_col]].copy()
            view[overview_score_col] = pd.to_numeric(view[overview_score_col], errors="coerce")
            view = view.dropna().sort_values(overview_score_col, ascending=False)
            view[overview_name_col] = view[overview_name_col].map(pretty_crop)
            view.columns = ["Crop", "Preliminary score"]
            view["Preliminary score"] = view["Preliminary score"].map(lambda x: fmt(x))
            st.dataframe(view, hide_index=True, use_container_width=True)
        else:
            st.info("Crop screening results are not available.")
    st.info(
        "Climate data can be requested for a selected global point and currently use 2025 observations. "
        "The series is not a long-term climate normal or a weather forecast. Soil and crop analyses still use the Antalya pilot dataset."
    )

elif page == "Climate Analysis":
    hero(
        "Climate analysis",
        "Monthly climate indicators derived from NASA POWER daily observations.",
    )
    selected_is_current = (
        not st.session_state.selected_climate.empty
        and st.session_state.climate_location == (
            st.session_state.selected_lat, st.session_state.selected_lon
        )
    )
    if selected_is_current:
        climate_to_show = st.session_state.selected_climate.copy()
        st.success(
            f"Showing NASA POWER data for selected coordinates: "
            f"{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}"
        )
    else:
        climate_to_show = climate.copy()
        st.info(
            "No newly fetched climate data for the selected point yet. "
            "Go to Overview, select a point on the world map, and click "
            "'Load climate data for this location'. The table below is the original Antalya pilot dataset."
        )

    if climate_to_show.empty:
        st.warning("Climate data are not available.")
    else:
        month = col(climate_to_show, ["year_month", "month", "date"])
        temp = col(climate_to_show, ["avg_temperature", "average_temperature", "mean_temperature"])
        rain = col(climate_to_show, ["total_rainfall", "total_rainfall_mm", "rainfall_mm"])
        solar = col(climate_to_show, ["avg_solar_radiation", "average_solar_radiation", "solar_radiation"])
        df = climate_to_show.copy()
        x = month if month else df.index
        if month:
            df[month] = df[month].astype(str)
            df = df.sort_values(month)
            if df[month].str.match(r"\d{4}-\d{2}").all():
                df["_month_label"] = pd.to_datetime(df[month] + "-01").dt.strftime("%b %Y")
                x = "_month_label"

        chart_specs = [
            ("Monthly average temperature", temp, "line", "#23734d", "Temperature (°C)"),
            ("Monthly total rainfall", rain, "bar", "#4b9b79", "Rainfall (mm)"),
            ("Monthly average solar radiation", solar, "bar", "#d9a441", "Solar radiation (kWh/m²/day)"),
        ]
        for title, y, chart_type, color, axis_title in chart_specs:
            if y:
                df[y] = pd.to_numeric(df[y], errors="coerce")
                section(title)
                if chart_type == "line":
                    fig = px.line(df, x=x, y=y, markers=True, labels={y: axis_title, x: "Month"})
                    fig.update_traces(line_color=color, line_width=3, marker_size=7)
                else:
                    fig = px.bar(df, x=x, y=y, labels={y: axis_title, x: "Month"},
                                 color_discrete_sequence=[color])
                fig.update_layout(template="plotly_white", height=320, margin=dict(l=15, r=15, t=20, b=15))
                st.plotly_chart(fig, use_container_width=True)
        with st.expander("View monthly source data"):
            st.dataframe(climate_to_show, use_container_width=True, hide_index=True)
    st.caption("Source: NASA POWER daily point API. Selected-location data are aggregated to monthly values for 2025.")

elif page == "Soil Analysis":
    hero(
        "Soil analysis",
        "Location-specific soil properties are shown only when data are available for the selected point.",
    )
    is_antalla_pilot = (
        abs(st.session_state.selected_lat - DEFAULT_LAT) < 0.02
        and abs(st.session_state.selected_lon - DEFAULT_LON) < 0.02
    )
    if is_antalla_pilot:
        st.info("The selected point is within the Antalya pilot area. The following profile is the existing SoilGrids estimate for the pilot coordinates.")
        if not soil.empty:
            section("Antalya pilot site summary")
            display_soil = soil.copy()
            display_soil.columns = [str(c).replace("_", " ").title() for c in display_soil.columns]
            st.dataframe(display_soil, use_container_width=True, hide_index=True)
        if not profile.empty:
            section("Soil profile by depth")
            depth = col(profile, ["depth_label", "depth"])
            texture_cols = [col(profile, x) for x in ["sand_pct", "silt_pct", "clay_pct"]]
            texture_cols = [x for x in texture_cols if x]
            if depth and texture_cols:
                long = profile[[depth] + texture_cols].melt(
                    id_vars=depth, var_name="Particle", value_name="Percentage"
                )
                long["Particle"] = long["Particle"].map({
                    "sand_pct": "Sand", "silt_pct": "Silt", "clay_pct": "Clay"
                }).fillna(long["Particle"])
                fig = px.bar(long, x="Percentage", y=depth, color="Particle",
                             orientation="h", barmode="stack",
                             title="Soil texture by depth (%)")
                fig.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(profile, hide_index=True, use_container_width=True)
    else:
        st.warning(
            "No verified location-specific soil profile is currently available for this selected point. "
            "SoilGrids REST point-query service is paused, and the WCS/COG integration has not yet been implemented. "
            "The Antalya soil profile is intentionally not displayed as if it belonged to this location."
        )
        st.markdown(
            "[SoilGrids data access documentation](https://docs.isric.org/globaldata/soilgrids/wcs.html)"
        )
    st.caption(
        "SoilGrids provides gridded model estimates, not field measurements. "
        "A selected coordinate does not guarantee a field-level soil assessment."
    )

elif page == "Crop Suitability":
    hero(
        "Crop suitability screening",
        "Location-specific preliminary screening using NASA POWER climate data and FAO ECOCROP crop requirements.",
    )
    location_current = (
        not st.session_state.selected_climate.empty
        and st.session_state.climate_location == (
            st.session_state.selected_lat, st.session_state.selected_lon
        )
    )
    if location_current and not st.session_state.location_suitability.empty:
        view = st.session_state.location_suitability.copy()
        st.success(
            f"Climate-based screening for {st.session_state.selected_lat:.4f}, "
            f"{st.session_state.selected_lon:.4f}"
        )
        section("Crop ranking for selected location")
        chart_data = view.dropna(subset=["Preliminary score"])
        if not chart_data.empty:
            fig = px.bar(
                chart_data, x="Crop", y="Preliminary score",
                text_auto=".1f", color="Preliminary score",
                color_continuous_scale=["#d7e8dc", "#1f704d"], range_color=(0, 100)
            )
            fig.update_layout(template="plotly_white", height=360,
                              coloraxis_showscale=False,
                              margin=dict(l=15, r=15, t=20, b=15))
            st.plotly_chart(fig, use_container_width=True)
        display = view.drop(columns=["crop_id"]).copy()
        for c in ["Temperature fit", "Annual rainfall fit", "Soil pH fit", "Preliminary score"]:
            display[c] = display[c].map(lambda x: "Not available" if pd.isna(x) else fmt(x))
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.warning(
            "These are climate-based preliminary screening scores, not yield predictions. "
            "Soil pH is not included because location-specific SoilGrids retrieval is not yet connected; "
            "Antalya soil values are deliberately not reused for other locations."
        )
    else:
        st.info(
            "Select a point on the Overview map and load its NASA POWER climate data to calculate "
            "location-specific crop screening. Until then, the database contains the original Antalya pilot assessment."
        )
        if not suit.empty:
            st.dataframe(suit, hide_index=True, use_container_width=True)

elif page == "Crop Rotation":
    hero(
        "Crop rotation scenarios",
        "Three-year candidate sequences recalculated from the selected location's climate-based crop screening.",
    )
    location_current = (
        not st.session_state.selected_climate.empty
        and st.session_state.climate_location == (
            st.session_state.selected_lat, st.session_state.selected_lon
        )
    )
    if location_current and not st.session_state.location_rotations.empty:
        st.success(
            f"Rotation candidates for {st.session_state.selected_lat:.4f}, "
            f"{st.session_state.selected_lon:.4f}"
        )
        rows = st.session_state.location_rotations
        for index, (_, row) in enumerate(rows.iterrows(), start=1):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(
                    f'<div class="rotation-card {"top" if index == 1 else ""}">'
                    f'<div class="rotation-sequence">#{index} &nbsp; '
                    f'{row["Year 1"]} → {row["Year 2"]} → {row["Year 3"]}</div>'
                    f'<div class="rotation-groups">{row["Crop group sequence"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with right:
                st.metric("Mean score", fmt(row["Mean screening score"]))
        st.warning(
            "Rotation scenarios are heuristic candidates based on climate-only preliminary scores. "
            "They do not account for field history, planting calendars, pests, nutrient balances, "
            "irrigation constraints, or farm economics. They are not agronomic prescriptions."
        )
    else:
        st.info(
            "Select a point on the Overview map and load its NASA POWER climate data to generate "
            "location-specific rotation candidates. The database rotation table is the original Antalya pilot."
        )
        if not rotation.empty:
            st.dataframe(rotation, hide_index=True, use_container_width=True)

elif page == "Data Sources":
    hero(
        "Data sources and methodology",
        "Review the datasets behind SmartCrop and understand the limits of each analysis.",
    )
    sources = [
        ("NASA POWER", "Climate data", "Daily meteorological and solar-energy parameters.", "https://power.larc.nasa.gov/"),
        ("ISRIC SoilGrids", "Soil data", "Gridded global predictions of selected soil properties.", "https://www.isric.org/explore/soilgrids"),
        ("FAO", "Agricultural references", "General references for crop water requirements and rooting depths.", "https://www.fao.org/"),
        ("FAO ECOCROP", "Crop requirements", "Crop ecological requirement reference database.", "https://ecocrop.apps.fao.org/"),
    ]
    for source_name, category, description, url in sources:
        with st.container(border=True):
            st.markdown(f"**{source_name}** · {category}")
            st.write(description)
            st.link_button("Open source", url)

    section("Methodology and important limitations")
    limitations = [
        "Selected-point climate data are fetched from NASA POWER for 2025; they are not a long-term climate normal or a weather forecast.",
        "NASA POWER grid resolution varies by parameter; nearby selected points may return the same grid-cell data.",
        "Global point selection fetches NASA POWER climate data. Crop screening and rotation are recalculated from climate data and ECOCROP requirements; soil pH is excluded unless location-specific soil data are available.",
        "SoilGrids values are gridded estimates and have not been validated with field soil samples.",
        "The Antalya pilot pH value is a nearest-valid-pixel estimate. It is not reused for other selected locations.",
        "Rainfall fit compares one observed calendar year's rainfall with ECOCROP reference ranges. It is not irrigation demand.",
        "For selected locations, the preliminary climate-only score is the equal-weighted mean of available temperature and annual-rainfall fits; soil pH is excluded until location-specific soil data are connected.",
        "Root-zone water storage is estimated using generalized rooting-depth references and SoilGrids layer values.",
        "Root-zone storage is not plant water uptake at a particular date and is not an irrigation prescription.",
        "The rotation generator does not model field history, planting calendars, pests, diseases, nutrient balances, soil carbon, irrigation, or economics.",
        "Missing crop requirement data remain missing; they are not converted to zero.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")

    st.warning(
        "SmartCrop is an experimental decision-support prototype. "
        "Its screening results should not replace local agronomic advice or field measurements."
    )

st.markdown(
    '<div class="footer">SmartCrop · NASA Space Apps Challenge · Antalya pilot study</div>',
    unsafe_allow_html=True,
)
