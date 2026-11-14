import sqlite3
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

if page == "Overview":
    hero(
        "Smarter decisions for healthier fields.",
        "Explore satellite-derived climate and soil information, compare crop requirements, and review transparent rotation scenarios.",
    )
    score_col = col(suit, ["preliminary_screening_score", "preliminary_score"])
    valid_scores = (
        pd.to_numeric(suit[score_col], errors="coerce").notna().sum()
        if score_col else 0
    )
    a, b, c, d = st.columns(4)
    a.metric("Selected point", "Global")
    b.metric("Climate records", f"{len(climate)} months" if not climate.empty else "Unavailable")
    c.metric("Scored crops", str(valid_scores))
    d.metric("Rotation candidates", str(len(rotation)))

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
                st.success("NASA POWER climate data loaded for the selected point.")
            except Exception as exc:
                st.error(f"NASA POWER data could not be loaded: {exc}")
        st.caption("Global point selection is enabled. Soil and crop results below still use the Antalya pilot dataset.")
    with right:
        section("Preliminary crop screening")
        name = col(suit, ["crop_name_en", "crop_name", "name_en", "crop_id"])
        if score_col and name and not suit.empty:
            view = suit[[name, score_col]].copy()
            view[score_col] = pd.to_numeric(view[score_col], errors="coerce")
            view = view.dropna().sort_values(score_col, ascending=False)
            view[name] = view[name].map(pretty_crop)
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
        "Explore soil texture by depth and estimated available-water storage across the soil profile.",
    )
    if not soil.empty:
        section("Site summary")
        display_soil = soil.copy()
        display_soil.columns = [str(c).replace("_", " ").title() for c in display_soil.columns]
        st.dataframe(display_soil, use_container_width=True, hide_index=True)
    else:
        st.info("The soil_site_summary table is not available.")

    if not profile.empty:
        section("Soil profile by depth")
        depth = col(profile, ["depth_label", "depth"])
        texture_cols = [col(profile, [x]) for x in ["sand_pct", "silt_pct", "clay_pct"]]
        texture_cols = [x for x in texture_cols if x]
        if depth and texture_cols:
            long = profile[[depth] + texture_cols].melt(
                id_vars=depth, var_name="Particle", value_name="Percentage"
            )
            long["Particle"] = long["Particle"].map({
                "sand_pct": "Sand", "silt_pct": "Silt", "clay_pct": "Clay"
            }).fillna(long["Particle"])
            fig = px.bar(
                long, x="Percentage", y=depth, color="Particle", orientation="h",
                barmode="stack", title="Soil texture by depth (%)",
                color_discrete_sequence=["#d8b56d", "#9ab9a4", "#647e69"],
            )
            fig.update_layout(template="plotly_white", height=360, margin=dict(l=15, r=15, t=45, b=15))
            st.plotly_chart(fig, use_container_width=True)

        aw = col(profile, ["available_water_mm"])
        if aw:
            section("Estimated available water by soil layer")
            tmp = profile.copy()
            tmp[aw] = pd.to_numeric(tmp[aw], errors="coerce")
            fig = px.bar(
                tmp, x=depth if depth else tmp.index, y=aw,
                labels={aw: "Available-water storage (mm)", depth: "Soil depth"} if depth else {aw: "Available-water storage (mm)"},
                color_discrete_sequence=["#3f8c70"],
            )
            fig.update_layout(template="plotly_white", height=300, margin=dict(l=15, r=15, t=20, b=15))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View complete soil profile records"):
            st.dataframe(profile, use_container_width=True, hide_index=True)

    if not root.empty:
        water = col(root, ["root_zone_storage_mm"])
        crop = col(root, ["crop_id"])
        if water and crop:
            section("Estimated root-zone water storage by crop")
            tmp = root[[crop, water]].copy()
            tmp[crop] = tmp[crop].map(pretty_crop)
            tmp[water] = pd.to_numeric(tmp[water], errors="coerce")
            fig = px.bar(
                tmp, x=crop, y=water,
                labels={crop: "Crop", water: "Estimated storage (mm)"},
                color_discrete_sequence=["#28734f"],
            )
            fig.update_layout(template="plotly_white", height=300, margin=dict(l=15, r=15, t=20, b=15))
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "SoilGrids values are gridded estimates, not field measurements. "
        "Root-zone storage is not the same as irrigation demand or water available to the plant at a specific date."
    )

elif page == "Crop Suitability":
    hero(
        "Crop suitability screening",
        "An explainable preliminary screening based on annual rainfall fit and soil pH fit.",
    )
    if suit.empty:
        st.warning("The crop_suitability_assessment table is missing. Run the screening engine first.")
    else:
        name = col(suit, ["crop_name_en", "crop_name", "name_en", "crop_id"])
        rain = col(suit, ["rainfall_fit_score", "rainfall_score"])
        ph = col(suit, ["ph_fit_score", "ph_score"])
        total = col(suit, ["preliminary_screening_score", "preliminary_score"])
        water = col(suit, ["root_zone_available_water_mm", "root_zone_water_mm"])
        selected = [x for x in [name, rain, ph, total, water] if x]
        view = suit[selected].copy()
        rename = {}
        for old, new in [
            (name, "Crop"), (rain, "Rainfall fit score"), (ph, "Soil pH fit score"),
            (total, "Preliminary screening score"),
            (water, "Root-zone water storage (mm)"),
        ]:
            if old:
                rename[old] = new
        view = view.rename(columns=rename)
        if "Crop" in view:
            view["Crop"] = view["Crop"].map(pretty_crop)
        for c in view.columns:
            if c != "Crop":
                view[c] = pd.to_numeric(view[c], errors="coerce")

        if "Preliminary screening score" in view:
            view = view.sort_values("Preliminary screening score", ascending=False, na_position="last")
            valid = view.dropna(subset=["Preliminary screening score"])
            if not valid.empty:
                section("Preliminary screening ranking")
                fig = px.bar(
                    valid, x="Crop", y="Preliminary screening score",
                    text_auto=".2f", color="Preliminary screening score",
                    color_continuous_scale=["#d7e8dc", "#1f704d"], range_color=(0, 100),
                )
                fig.update_layout(template="plotly_white", height=360, coloraxis_showscale=False,
                                  margin=dict(l=15, r=15, t=20, b=15))
                st.plotly_chart(fig, use_container_width=True)

        section("Crop comparison")
        display = view.copy()
        for c in display.columns:
            if c != "Crop":
                display[c] = display[c].map(lambda x: "Not available" if pd.isna(x) else fmt(x))
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.info(
            "Screening score = (rainfall fit score + soil pH fit score) / 2. "
            "Both components must be available. Missing values are not treated as zero. "
            "This is a preliminary indicator, not a yield forecast or final planting recommendation."
        )

elif page == "Crop Rotation":
    hero(
        "Crop rotation scenarios",
        "Explore three-year candidate sequences generated using transparent diversity rules.",
    )
    if rotation.empty:
        st.warning("The crop_rotation_candidates table is missing. Run the rotation engine first.")
    else:
        score_name = col(rotation, ["mean_screening_score", "mean_score", "average_score"])
        section("Top rotation candidates")
        rows = rotation.copy()
        if score_name:
            rows[score_name] = pd.to_numeric(rows[score_name], errors="coerce")
            rows = rows.sort_values(score_name, ascending=False, na_position="last")

        crop_fields = []
        for year in [1, 2, 3]:
            crop_fields.append(col(rows, [f"year_{year}_crop_name", f"year{year}_crop_name", f"year_{year}_crop"]))
        group_fields = [
            col(rows, [f"year_{year}_crop_group", f"year{year}_crop_group"])
            for year in [1, 2, 3]
        ]

        for index, (_, row) in enumerate(rows.head(15).iterrows()):
            crops = [pretty_crop(row.get(field)) if field else "Not available" for field in crop_fields]
            groups = [pretty_crop(row.get(field)) if field else "Not available" for field in group_fields]
            score_value = row.get(score_name) if score_name else None
            sequence = " → ".join(crops)
            group_line = " → ".join(groups)
            card_class = "rotation-card top" if index == 0 else "rotation-card"
            left, right = st.columns([4, 1])
            with left:
                st.markdown(
                    f'<div class="{card_class}"><div class="rotation-sequence">'
                    f'#{index + 1} &nbsp; {sequence}</div>'
                    f'<div class="rotation-groups">Crop groups: {group_line}</div></div>',
                    unsafe_allow_html=True,
                )
            with right:
                st.metric("Mean screening score", fmt(score_value) if score_value is not None else "Not available")

        st.info(
            "Rules: (1) no crop repeats within the three-year sequence; "
            "(2) the same crop group cannot occur in adjacent years. "
            "Ranking is the mean of the three preliminary crop screening scores. "
            "This is a heuristic candidate generator, not agronomic validation."
        )

    with st.expander("Excluded crops and model limitations"):
        st.markdown(
            "- Cotton and chickpea are excluded when complete rainfall and pH screening scores are unavailable.\n"
            "- Olive is perennial and is not included in annual rotation sequences.\n"
            "- Field crop history, local planting calendars, pest and disease cycles, nutrient balances, "
            "irrigation constraints, and farm economics are not modelled.\n"
            "- Different adjacent crop groups do not guarantee a beneficial rotation.\n"
            "- Mean screening score is not a prediction of rotation yield or performance."
        )

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
        "The current interactive global selection updates climate data only. Soil, crop suitability, and rotation tables remain based on the Antalya pilot dataset until location-specific soil data are connected.",
        "SoilGrids values are gridded estimates and have not been validated with field soil samples.",
        "The selected pH value is a nearest-valid-pixel estimate; its pixel distance is not recorded in the current site summary.",
        "Rainfall fit compares one observed calendar year's rainfall with ECOCROP reference ranges. It is not irrigation demand.",
        "The preliminary screening score is the equal-weighted mean of rainfall fit and soil pH fit.",
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
