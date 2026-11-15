import sqlite3
import itertools
import math
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



@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasa_power_climate_normals(latitude, longitude, start_year=1991, end_year=2020):
    """Fetch monthly NASA POWER records and derive a 30-year monthly reference."""
    endpoint = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": round(float(longitude), 4),
        "latitude": round(float(latitude), 4),
        "start": int(start_year),
        "end": int(end_year),
        "format": "JSON",
    }
    response = requests.get(endpoint, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        raise ValueError("; ".join(str(item) for item in messages))

    parameter_data = payload.get("properties", {}).get("parameter", {})
    required = ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
    missing = [name for name in required if name not in parameter_data]
    if missing:
        raise ValueError("NASA POWER monthly response is missing: " + ", ".join(missing))

    series_list = []
    for parameter in required:
        series = pd.Series(parameter_data[parameter], name=parameter)
        series.index = pd.to_datetime(series.index.astype(str), format="%Y%m", errors="coerce")
        series = pd.to_numeric(series, errors="coerce").replace(-999, float("nan"))
        series_list.append(series)

    history = pd.concat(series_list, axis=1).sort_index()
    history = history.loc[~history.index.isna()].dropna(how="all")
    if history.empty:
        raise ValueError("NASA POWER returned no valid monthly observations.")

    history = history.reset_index().rename(columns={
        "index": "month_date",
        "T2M": "avg_temperature",
        "PRECTOTCORR": "mean_daily_rainfall",
        "ALLSKY_SFC_SW_DWN": "avg_solar_radiation",
    })
    if "month_date" not in history.columns:
        history = history.rename(columns={history.columns[0]: "month_date"})

    history["year"] = history["month_date"].dt.year
    history["month_number"] = history["month_date"].dt.month
    history["days_in_month"] = history["month_date"].dt.days_in_month
    history["year_month"] = history["month_date"].dt.strftime("%Y-%m")

    for column in ["avg_temperature", "mean_daily_rainfall", "avg_solar_radiation"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
        history.loc[history[column] <= -900, column] = float("nan")

    # NASA POWER monthly precipitation is a mean daily rate (mm/day).
    history["total_rainfall"] = history["mean_daily_rainfall"] * history["days_in_month"]

    valid_years = history.dropna(
        subset=["avg_temperature", "total_rainfall"]
    )["year"].nunique()
    if valid_years < 25:
        raise ValueError(
            f"Only {valid_years} valid years were returned; at least 25 are required."
        )

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    normals = (
        history.groupby("month_number", as_index=False)
        .agg(
            avg_temperature=("avg_temperature", "mean"),
            total_rainfall=("total_rainfall", "mean"),
            avg_solar_radiation=("avg_solar_radiation", "mean"),
            days_in_month=("days_in_month", "mean"),
            years_available=("year", "nunique"),
        )
        .sort_values("month_number")
    )
    normals["month"] = normals["month_number"].map(month_names)
    normals["year_month"] = normals["month"]
    normals["mean_daily_rainfall"] = normals["total_rainfall"] / normals["days_in_month"]

    annual_rows = []
    for year, group in history.groupby("year"):
        valid_temp = group["avg_temperature"].notna()
        weighted_temp = (
            (group.loc[valid_temp, "avg_temperature"] * group.loc[valid_temp, "days_in_month"]).sum()
            / group.loc[valid_temp, "days_in_month"].sum()
            if valid_temp.any() else float("nan")
        )
        annual_rows.append({
            "year": int(year),
            "annual_mean_temperature": weighted_temp,
            "annual_total_rainfall": group["total_rainfall"].sum(min_count=1),
            "annual_mean_solar_radiation": group["avg_solar_radiation"].mean(),
            "valid_months": int(
                group[["avg_temperature", "total_rainfall"]].notna().all(axis=1).sum()
            ),
        })
    annual = pd.DataFrame(annual_rows).sort_values("year")

    if len(normals) != 12:
        raise ValueError(f"Expected 12 monthly normals, received {len(normals)}.")

    return {
        "monthly_normals": normals,
        "monthly_history": history,
        "annual_summary": annual,
        "period": f"{start_year}–{end_year}",
        "valid_years": int(valid_years),
    }



@st.cache_data(ttl=21600, show_spinner=False)
def fetch_soilgrids_point(latitude, longitude):
    """Fetch 0–5 cm SoilGrids mean estimates via WCS.

    WCS responses are validated by opening the raster itself, not by relying
    on the HTTP Content-Type or a narrow TIFF magic-byte check. The server may
    label a valid GeoTIFF response as image/tiff.
    """
    try:
        import numpy as np
        import rasterio
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "SoilGrids dependencies are missing. Install rasterio, pyproj, and numpy."
        ) from exc

    latitude, longitude = float(latitude), float(longitude)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordinates are outside valid latitude/longitude bounds.")

    soilgrids_proj = "+proj=igh +datum=WGS84 +no_defs +towgs84=0,0,0"
    transformer = Transformer.from_crs("EPSG:4326", soilgrids_proj, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("Could not project this coordinate into the SoilGrids grid.")

    # Request a compact 1 km x 1 km window (native pixels are ~250 m).
    # Smaller WCS payloads reduce server processing and transfer time.
    half_window = 500.0
    subsets = [
        ("SUBSET", f"X({x-half_window:.2f},{x+half_window:.2f})"),
        ("SUBSET", f"Y({y-half_window:.2f},{y+half_window:.2f})"),
    ]
    crs_uri = "http://www.opengis.net/def/crs/EPSG/0/152160"
    properties = {"phh2o": "Soil pH", "sand": "Sand", "silt": "Silt", "clay": "Clay"}
    rasters, failures, distances = {}, {}, {}

    def get_raster(prop):
        endpoint = f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
        params = [
            ("SERVICE", "WCS"), ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"), ("COVERAGEID", f"{prop}_0-5cm_mean"),
            ("FORMAT", "GEOTIFF_INT16"), *subsets,
            ("SUBSETTINGCRS", crs_uri), ("OUTPUTCRS", crs_uri),
            ("RESX", "250"), ("RESY", "250"),
        ]
        response = requests.get(
            endpoint,
            params=params,
            headers={"User-Agent": "SmartCrop/1.2 (SoilGrids WCS client)"},
            timeout=(10, 60),
        )
        response.raise_for_status()
        content = response.content
        if not content:
            raise RuntimeError("SoilGrids returned an empty response.")

        # Do not reject by Content-Type or TIFF signature: valid TIFF/GeoTIFF
        # payloads can be labelled image/tiff and may use different TIFF variants.
        try:
            with rasterio.io.MemoryFile(content) as memfile:
                with memfile.open() as ds:
                    if ds.driver not in ("GTiff", "COG"):
                        raise RuntimeError(
                            f"Unsupported raster driver returned by WCS: {ds.driver}"
                        )
                    if ds.count < 1 or ds.width < 1 or ds.height < 1:
                        raise RuntimeError("WCS returned an empty raster.")
                    arr = ds.read(1, masked=True)
                    values = np.asarray(arr.filled(np.nan), dtype=float)
                    valid = np.isfinite(values) & (ds.read_masks(1) > 0)
                    if ds.nodata is not None:
                        valid &= values != float(ds.nodata)
                    if prop == "phh2o":
                        valid &= (values >= 30) & (values <= 110)
                    else:
                        # Zero is a valid fraction; only NoData and out-of-range values are rejected.
                        valid &= (values >= 0) & (values <= 1000)
                    return values.copy(), valid.copy(), ds.transform, ds.crs
        except Exception as exc:
            content_type = response.headers.get("Content-Type", "unknown")
            preview = content[:120].hex(" ")
            raise RuntimeError(
                f"Could not open WCS response as a GeoTIFF (HTTP {response.status_code}, "
                f"Content-Type: {content_type}, bytes: {preview}): {exc}"
            ) from exc

    # Parallel requests keep the overall wait bounded by the slowest layer.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_prop = {executor.submit(get_raster, prop): prop for prop in properties}
        for future in as_completed(future_to_prop):
            prop = future_to_prop[future]
            try:
                rasters[prop] = future.result()
            except Exception as exc:
                failures[prop] = f"{type(exc).__name__}: {exc}"

    results = {key: None for key in properties}
    texture_props = ("sand", "silt", "clay")
    if all(prop in rasters for prop in texture_props):
        base_values, base_valid, base_transform, base_crs = rasters["sand"]
        common = base_valid.copy()
        aligned = True
        for prop in ("silt", "clay"):
            vals, mask, transform, crs = rasters[prop]
            if vals.shape != base_values.shape or transform != base_transform or crs != base_crs:
                aligned = False
                break
            common &= mask
        if aligned and common.any():
            rows, cols = np.where(common)
            pixel_distances = []
            for row, col_idx in zip(rows, cols):
                px, py = rasterio.transform.xy(base_transform, int(row), int(col_idx), offset="center")
                pixel_distances.append(math.hypot(px-x, py-y))
            best = int(np.argmin(pixel_distances))
            row, col_idx = int(rows[best]), int(cols[best])
            for prop in texture_props:
                results[prop] = float(rasters[prop][0][row, col_idx]) / 10.0
                distances[prop] = round(float(pixel_distances[best]), 1)
        elif not aligned:
            failures["texture"] = "SoilGrids texture rasters were not aligned; no mixed-cell composition was calculated."
        else:
            failures["texture"] = "No common valid pixel for sand, silt and clay in the requested neighbourhood."
    else:
        missing = [prop for prop in texture_props if prop not in rasters]
        failures["texture"] = "Unavailable texture layer(s): " + ", ".join(missing)

    if "phh2o" in rasters:
        values, valid, transform, _ = rasters["phh2o"]
        if valid.any():
            rows, cols = np.where(valid)
            dists = []
            for row, col_idx in zip(rows, cols):
                px, py = rasterio.transform.xy(transform, int(row), int(col_idx), offset="center")
                dists.append(math.hypot(px-x, py-y))
            best = int(np.argmin(dists))
            results["phh2o"] = float(values[rows[best], cols[best]]) / 10.0
            distances["phh2o"] = round(float(dists[best]), 1)
        else:
            failures["phh2o"] = "Raster opened, but no valid pH pixels were present in the requested window."

    if all(results.get(k) is None for k in properties):
        detail = "; ".join(f"{properties.get(k, k)}: {v}" for k, v in failures.items())
        raise ValueError("SoilGrids WCS returned no usable estimates. " + detail)

    texture_complete = all(results.get(k) is not None for k in texture_props)
    if texture_complete:
        texture_sum = sum(results[k] for k in texture_props)
        if 85.0 <= texture_sum <= 115.0:
            results["texture_sum_pct"] = texture_sum
        else:
            for prop in texture_props:
                results[prop] = None
                distances[prop] = None
            results["texture_sum_pct"] = None
            failures["texture"] = (
                f"Texture fractions totalled {texture_sum:.1f}%, outside the 85–115% validation range."
            )
            texture_complete = False
    else:
        results["texture_sum_pct"] = None

    results["sample_distance_m"] = distances
    results["property_errors"] = failures
    results["depth"] = "0–5 cm"
    results["source"] = "ISRIC SoilGrids 2.0 WCS"
    results["quantile"] = "Mean"
    results["texture_complete"] = texture_complete
    return results



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


def classify_usda_texture(sand, silt, clay):
    """Classify mineral-soil texture from sand/silt/clay percentages.

    This deterministic USDA-triangle approximation is used only to map the
    broad ECOCROP texture descriptors (light/medium/heavy), not as a lab result.
    """
    values = [sand, silt, clay]
    if any(v is None or pd.isna(v) for v in values):
        return None
    sand, silt, clay = [max(0.0, float(v)) for v in values]
    total = sand + silt + clay
    if not (70.0 <= total <= 130.0):
        return None
    sand, silt, clay = [v * 100.0 / total for v in (sand, silt, clay)]

    if sand >= 85 and (silt + 1.5 * clay) < 15:
        return "sand"
    if sand >= 70 and (silt + 2.0 * clay) < 30:
        return "loamy sand"
    if clay >= 40:
        if silt >= 40:
            return "silty clay"
        if sand >= 45:
            return "sandy clay"
        return "clay"
    if clay >= 27:
        if silt >= 40 and sand <= 20:
            return "silty clay loam"
        if sand > 45:
            return "sandy clay loam"
        return "clay loam"
    if clay >= 20:
        if sand > 52:
            return "sandy clay loam"
        if silt >= 50:
            return "silt loam"
        return "clay loam"
    if clay >= 7:
        if sand > 52:
            return "sandy loam"
        if silt >= 50:
            return "silt loam"
        return "loam"
    if silt >= 80:
        return "silt"
    if silt >= 50:
        return "silt loam"
    if sand >= 70:
        return "loamy sand"
    if sand > 52:
        return "sandy loam"
    return "loam"


def texture_group(texture_class):
    if texture_class in {"sand", "loamy sand", "sandy loam"}:
        return "light"
    if texture_class in {"loam", "silt loam", "silt"}:
        return "medium"
    if texture_class in {
        "sandy clay loam", "clay loam", "silty clay loam",
        "sandy clay", "silty clay", "clay"
    }:
        return "heavy"
    return None


def texture_fit_score(texture_class, optimal_textures, absolute_textures):
    """Score broad texture compatibility without inventing missing ECOCROP data."""
    if not texture_class:
        return None
    actual_group = texture_group(texture_class)
    if not actual_group:
        return None

    def parse_groups(raw):
        if raw is None or pd.isna(raw):
            return set()
        return {part.strip().lower() for part in str(raw).split(",") if part.strip()}

    optimal = parse_groups(optimal_textures)
    absolute = parse_groups(absolute_textures)
    if "wide" in optimal or "wide" in absolute:
        return 100.0
    # ECOCROP uses qualitative categories; 'organic' is not a mineral texture class.
    if actual_group in optimal:
        return 100.0
    if actual_group in absolute:
        return 60.0
    if absolute:
        return 0.0
    return None


def build_location_suitability(climate_df, ecological, catalog, soil_ph=None, soil_texture=None):
    """Location screening using 1991–2020 climate normals and available soil estimates."""
    if climate_df.empty or ecological.empty:
        return pd.DataFrame()

    temp_col = col(climate_df, ["avg_temperature"])
    rain_col = col(climate_df, ["total_rainfall"])
    if not temp_col or not rain_col:
        return pd.DataFrame()

    temperature_values = pd.to_numeric(climate_df[temp_col], errors="coerce")
    rainfall_values = pd.to_numeric(climate_df[rain_col], errors="coerce")
    days_col = col(climate_df, ["days_in_month"])
    if days_col:
        day_weights = pd.to_numeric(climate_df[days_col], errors="coerce")
        valid_temp = temperature_values.notna() & day_weights.notna()
        annual_temp = (
            (temperature_values[valid_temp] * day_weights[valid_temp]).sum()
            / day_weights[valid_temp].sum()
            if valid_temp.any() and day_weights[valid_temp].sum() > 0 else float("nan")
        )
    else:
        annual_temp = temperature_values.mean()
    annual_rain = rainfall_values.sum(min_count=1)
    ph_available = soil_ph is not None and not pd.isna(soil_ph)

    texture_class = None
    if soil_texture:
        texture_class = classify_usda_texture(
            soil_texture.get("sand"), soil_texture.get("silt"), soil_texture.get("clay")
        )

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
        if ph_available:
            ph_score = range_fit(
                float(soil_ph), crop.get("ph_opt_min"), crop.get("ph_opt_max"),
                crop.get("ph_abs_min"), crop.get("ph_abs_max")
            )
        texture_score = texture_fit_score(
            texture_class, crop.get("texture_optimal"), crop.get("texture_absolute")
        ) if texture_class else None

        critical_scores = [temp_score, rain_score]
        if ph_available:
            critical_scores.append(ph_score)
        if texture_score is not None:
            critical_scores.append(texture_score)
        critical_data_complete = all(v is not None for v in critical_scores)
        outside_absolute_limit = (
            critical_data_complete and any(float(v) <= 0 for v in critical_scores)
        )

        # Transparent weights: temperature 30%, annual rainfall 25%, pH 20%,
        # texture 25%. Missing components are excluded and remaining weights renormalized.
        weighted_parts = []
        for value, weight in [
            (temp_score, 0.30), (rain_score, 0.25),
            (ph_score, 0.20), (texture_score, 0.25),
        ]:
            if value is not None and not pd.isna(value):
                weighted_parts.append((float(value), weight))
        weight_total = sum(weight for _, weight in weighted_parts)
        weighted_score = (
            sum(value * weight for value, weight in weighted_parts) / weight_total
            if weight_total else None
        )
        # Keep the weighted screening score visible even when a crop crosses
        # an absolute threshold. The threshold is represented separately in
        # Eligibility and rotation filtering, rather than erasing all other
        # useful component information by forcing the composite score to 0.
        score = weighted_score

        cat = catalog[catalog["crop_id"] == crop["crop_id"]]
        crop_name = cat.iloc[0]["crop_name_en"] if not cat.empty else str(crop["crop_id"]).title()
        crop_group = cat.iloc[0]["crop_group"] if not cat.empty else "Unknown"
        if outside_absolute_limit:
            eligibility = "Outside absolute limits"
        elif not critical_data_complete:
            eligibility = "Incomplete data"
        elif ph_available and texture_score is not None:
            eligibility = "Climate + pH + texture screening"
        elif ph_available:
            eligibility = "Partial soil screening (pH only)"
        elif texture_score is not None:
            eligibility = "Partial soil screening (texture only)"
        else:
            eligibility = "Climate-only screening"

        rows.append({
            "crop_id": crop["crop_id"], "Crop": crop_name, "Crop group": crop_group,
            "Temperature fit": temp_score, "Annual rainfall fit": rain_score,
            "Soil pH fit": ph_score, "Soil texture class": texture_class,
            "Soil texture fit": texture_score, "Preliminary score": score,
            "Eligibility": eligibility,
            "Data components": sum(v is not None and not pd.isna(v) for v in
                                    [temp_score, rain_score, ph_score, texture_score]),
        })

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["Preliminary score", "Crop"], ascending=[False, True], na_position="last"
    )

def build_location_rotations(suitability):
    """
    Build three-year illustrative rotations from eligible annual crops only.
    Perennial crops (including olive) are excluded. Crops cannot repeat and
    adjacent years cannot use the same crop group.
    """
    if suitability.empty:
        return pd.DataFrame()

    candidates_source = suitability.copy()
    if "Eligibility" in candidates_source.columns:
        candidates_source = candidates_source[
            candidates_source["Eligibility"].isin(
                [
                    "Eligible for screening",
                    "Partial soil screening",
                    "Partial soil screening (pH only)",
                    "Partial soil screening (texture only)",
                    "Climate + pH + texture screening",
                    "Climate-only screening",
                ]
            )
        ]

    # Explicit perennial exclusions. These crops belong in orchard planning,
    # not annual field-rotation sequences.
    perennial_ids = {"olive"}
    perennial_groups = {"fruit", "perennial", "tree crop", "orchard"}
    candidates_source = candidates_source[
        ~candidates_source["crop_id"].astype(str).str.lower().isin(perennial_ids)
        & ~candidates_source["Crop group"].astype(str).str.lower().isin(perennial_groups)
    ]

    valid = candidates_source.dropna(subset=["Preliminary score"]).to_dict("records")
    if len(valid) < 3:
        return pd.DataFrame()

    candidates = []
    for a, b, c in itertools.permutations(valid, 3):
        ids = [str(a["crop_id"]), str(b["crop_id"]), str(c["crop_id"])]
        groups = [str(a["Crop group"]), str(b["Crop group"]), str(c["Crop group"])]
        if len(set(ids)) != 3:
            continue
        # A three-year sequence must rotate across three distinct crop groups.
        # This prevents cereal -> oilseed -> cereal from being ranked as a diverse rotation.
        if len(set(group.lower() for group in groups)) != 3:
            continue

        scores = [float(a["Preliminary score"]), float(b["Preliminary score"]), float(c["Preliminary score"])]
        # Deterministic ordering for reproducible results.
        candidates.append({
            "Year 1": a["Crop"],
            "Year 2": b["Crop"],
            "Year 3": c["Crop"],
            "Crop group sequence": f'{groups[0]} → {groups[1]} → {groups[2]}',
            "Mean screening score": sum(scores) / 3,
            "Sequence key": " | ".join(ids),
        })

    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame(candidates).sort_values(
        ["Mean screening score", "Sequence key"],
        ascending=[False, True],
        kind="mergesort",
    ).head(15)
    return result.drop(columns=["Sequence key"]).reset_index(drop=True)


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
if "selected_climate_history" not in st.session_state:
    st.session_state.selected_climate_history = pd.DataFrame()
if "selected_annual_climate" not in st.session_state:
    st.session_state.selected_annual_climate = pd.DataFrame()
if "climate_period" not in st.session_state:
    st.session_state.climate_period = None
if "climate_location" not in st.session_state:
    st.session_state.climate_location = None
if "location_suitability" not in st.session_state:
    st.session_state.location_suitability = pd.DataFrame()
if "location_rotations" not in st.session_state:
    st.session_state.location_rotations = pd.DataFrame()
if "location_soil_ph" not in st.session_state:
    st.session_state.location_soil_ph = None
if "selected_soil" not in st.session_state:
    st.session_state.selected_soil = None
if "soil_location" not in st.session_state:
    st.session_state.soil_location = None

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
    # Only show location-specific screening after climate data have been loaded
    # for the exact currently selected coordinates. Never fall back to the
    # static Antalya pilot scores for an unrelated global point.
    if overview_location_current and not st.session_state.location_suitability.empty:
        overview_scores = st.session_state.location_suitability.copy()
        overview_name_col = "Crop"
        overview_score_col = "Preliminary score"
        overview_rotation_count = len(st.session_state.location_rotations)
        overview_climate_count = len(st.session_state.selected_climate)
        screening_calculated = True
    else:
        overview_scores = pd.DataFrame()
        overview_name_col = None
        overview_score_col = None
        overview_rotation_count = 0
        overview_climate_count = 0
        screening_calculated = False

    valid_scores = (
        pd.to_numeric(overview_scores[overview_score_col], errors="coerce").notna().sum()
        if overview_score_col and not overview_scores.empty else 0
    )
    a, b, c, d = st.columns(4)
    a.metric("Selected point", "Global")
    b.metric("Climate records", f"{overview_climate_count} months" if screening_calculated else "Not loaded")
    c.metric("Scored crops", str(valid_scores) if screening_calculated else "Not calculated")
    d.metric(
        "Rotation candidates",
        (str(overview_rotation_count) if overview_rotation_count > 0 else "None eligible")
        if screening_calculated else "Not calculated",
    )

    left, right = st.columns([1.12, 1])
    with left:
        section("Select any location worldwide")
        st.caption("Click anywhere on the map to choose a point. Then load its NASA POWER climate data.")
        map_view = folium.Map(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            zoom_start=2 if st.session_state.climate_location is None else 5,
            tiles=None,
            min_zoom=1,
            max_zoom=18,
            control_scale=True,
        )
        # Esri World Street Map generally uses English cartographic labels,
        # unlike the default OSM layer which may show local-language names.
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Sources: Esri, Garmin, USGS, NOAA",
            name="World map (English labels)",
            overlay=False,
            control=False,
            show=True,
        ).add_to(map_view)
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
                st.session_state.selected_climate_history = pd.DataFrame()
                st.session_state.selected_annual_climate = pd.DataFrame()
                st.session_state.climate_period = None
                st.session_state.climate_location = None
                st.session_state.location_suitability = pd.DataFrame()
                st.session_state.location_rotations = pd.DataFrame()
                st.session_state.location_soil_ph = None
                st.session_state.selected_soil = None
                st.session_state.soil_location = None
                st.rerun()

        st.write(
            f"**Selected coordinates:** {st.session_state.selected_lat:.4f}, "
            f"{st.session_state.selected_lon:.4f}"
        )
        if st.button("Load climate data for this location", type="primary", use_container_width=True):
            try:
                with st.spinner(
                    "Requesting 30 years of NASA POWER monthly climate data (1991–2020)..."
                ):
                    climate_package = fetch_nasa_power_climate_normals(
                        st.session_state.selected_lat,
                        st.session_state.selected_lon,
                        1991,
                        2020,
                    )
                result = climate_package["monthly_normals"]
                st.session_state.selected_climate = result
                st.session_state.selected_climate_history = climate_package["monthly_history"]
                st.session_state.selected_annual_climate = climate_package["annual_summary"]
                st.session_state.climate_period = climate_package["period"]
                st.session_state.climate_location = (
                    st.session_state.selected_lat, st.session_state.selected_lon
                )
                # Never reuse soil values from a different coordinate. SoilGrids WCS
                # is queried separately on the Soil Analysis page.
                matching_soil = (
                    st.session_state.selected_soil
                    if st.session_state.soil_location == (
                        st.session_state.selected_lat, st.session_state.selected_lon
                    ) else None
                )
                selected_ph = matching_soil.get("phh2o") if matching_soil else None
                st.session_state.location_soil_ph = selected_ph
                st.session_state.location_suitability = build_location_suitability(
                    result, ecological, catalog, soil_ph=selected_ph,
                    soil_texture=matching_soil
                )
                st.session_state.location_rotations = build_location_rotations(
                    st.session_state.location_suitability
                )
                st.success("NASA POWER 30-year climate reference data loaded. Crop screening and rotation scenarios were recalculated using the 1991–2020 climate baseline and available ECOCROP requirements.")
            except Exception as exc:
                st.error(f"NASA POWER data could not be loaded: {exc}")
        st.caption("Global point selection is enabled. Climate screening uses a 1991–2020 NASA POWER reference period; SoilGrids data are retrieved separately for the selected point.")
    with right:
        section("Preliminary crop screening")
        if overview_location_current and overview_score_col and overview_name_col:
            st.caption(
                f"Live climate/soil screening · "
                f"{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}"
            )
        else:
            st.caption(
                "Original Antalya pilot assessment · load climate data for the selected point "
                "to refresh this table"
            )
        if screening_calculated and overview_score_col and overview_name_col and not overview_scores.empty:
            overview_columns = [overview_name_col, overview_score_col]
            if "Eligibility" in overview_scores.columns:
                overview_columns.append("Eligibility")
            view = overview_scores[overview_columns].copy()
            view[overview_score_col] = pd.to_numeric(view[overview_score_col], errors="coerce")
            view = view.dropna(subset=[overview_score_col]).sort_values(overview_score_col, ascending=False)
            view[overview_name_col] = view[overview_name_col].map(pretty_crop)
            rename_map = {overview_name_col: "Crop", overview_score_col: "Preliminary score"}
            view = view.rename(columns=rename_map)
            view["Preliminary score"] = view["Preliminary score"].map(lambda x: fmt(x))
            st.dataframe(view, hide_index=True, use_container_width=True)
            if "Eligibility" in view.columns and view["Eligibility"].eq("Outside absolute limits").any():
                st.warning("Some crops cross at least one ECOCROP absolute climate/soil limit. Their weighted screening scores remain visible for transparency, but they are excluded from rotation candidates.")
            if overview_rotation_count == 0:
                st.info("No three-crop rotation currently meets the screening eligibility and crop-group diversity rules for this point.")
        else:
            st.info(
                "Crop scores have not been calculated for this point yet. "
                "Load NASA POWER climate data for the selected coordinates to calculate crop suitability and rotation candidates."
            )
    st.info(
        "Climate data can be requested for a selected global point and use a 1991–2020, 30-year reference period. "
        "The reference period is a historical climate baseline, not a forecast. NASA POWER grid resolution varies by parameter; SoilGrids data are retrieved separately for the selected point."
    )

elif page == "Climate Analysis":
    hero(
        "Climate analysis",
        "30-year monthly climate reference derived from NASA POWER monthly observations.",
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
            f"Showing NASA POWER {st.session_state.climate_period} reference data for "
            f"{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}"
        )
        annual_history = st.session_state.selected_annual_climate.copy()
    else:
        climate_to_show = pd.DataFrame()
        annual_history = pd.DataFrame()
        st.info(
            "No long-term climate reference has been loaded for the selected point yet. "
            "Go to Overview, select a point on the map, and load its NASA POWER climate data."
        )

    if climate_to_show.empty:
        st.warning("Long-term climate reference data are not available for this selected point.")
    else:
        month = "month"
        df = climate_to_show.copy()
        chart_specs = [
            ("30-year mean monthly temperature", "avg_temperature", "line", "#23734d", "Temperature (°C)"),
            ("Mean monthly precipitation total", "total_rainfall", "bar", "#4b9b79", "Precipitation (mm/month)"),
            ("Mean daily solar radiation by month", "avg_solar_radiation", "bar", "#d9a441", "Solar radiation (MJ/m²/day)"),
        ]
        for title, y, chart_type, color, axis_title in chart_specs:
            if y in df.columns:
                df[y] = pd.to_numeric(df[y], errors="coerce")
                section(title)
                if chart_type == "line":
                    fig = px.line(
                        df, x=month, y=y, markers=True,
                        labels={y: axis_title, month: "Calendar month"}
                    )
                    fig.update_traces(line_color=color, line_width=3, marker_size=7)
                else:
                    fig = px.bar(
                        df, x=month, y=y,
                        labels={y: axis_title, month: "Calendar month"},
                        color_discrete_sequence=[color],
                    )
                fig.update_layout(
                    template="plotly_white", height=320,
                    margin=dict(l=15, r=15, t=20, b=15),
                )
                st.plotly_chart(fig, use_container_width=True)

        if not annual_history.empty:
            section("Year-to-year climate variability within the reference period")
            annual_specs = [
                ("Annual mean temperature", "annual_mean_temperature", "Temperature (°C)", "#23734d"),
                ("Annual precipitation total", "annual_total_rainfall", "Precipitation (mm/year)", "#4b9b79"),
            ]
            for title, value_col, axis_title, color in annual_specs:
                if value_col in annual_history.columns:
                    fig = px.line(
                        annual_history, x="year", y=value_col, markers=True,
                        labels={value_col: axis_title, "year": "Year"},
                        title=title,
                    )
                    fig.update_traces(line_color=color, line_width=2, marker_size=5)
                    fig.update_layout(
                        template="plotly_white", height=300,
                        margin=dict(l=15, r=15, t=35, b=15),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with st.expander("View annual climate summaries (1991–2020)"):
                st.dataframe(annual_history, hide_index=True, use_container_width=True)

        with st.expander("View 12-month climate reference table"):
            st.dataframe(climate_to_show, hide_index=True, use_container_width=True)
        with st.expander("View underlying monthly observations (1991–2020)"):
            st.dataframe(
                st.session_state.selected_climate_history,
                hide_index=True,
                use_container_width=True,
            )

    st.caption(
        "Source: NASA POWER Monthly Point API. Reference period: 1991–2020. "
        "Monthly precipitation totals are derived from the returned mean daily precipitation rate "
        "multiplied by the number of calendar days in each month. This is a historical reference, "
        "not a forecast."
    )

elif page == "Soil Analysis":
    hero(
        "Soil analysis",
        "Location-specific topsoil estimates retrieved from ISRIC SoilGrids.",
    )
    selected_point = (
        st.session_state.selected_lat, st.session_state.selected_lon
    )
    soil_is_current = (
        st.session_state.selected_soil is not None
        and st.session_state.soil_location == selected_point
    )

    st.write(
        f"**Selected coordinates:** {selected_point[0]:.4f}, {selected_point[1]:.4f}"
    )
    st.caption(
        "Retrieves SoilGrids mean predictions for the 0–5 cm topsoil layer "
        "at the selected map coordinate."
    )

    if st.button(
        "Load SoilGrids data for this location",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Requesting pH, sand, silt, and clay estimates from ISRIC SoilGrids..."
            ):
                soil_result = fetch_soilgrids_point(*selected_point)
            st.session_state.selected_soil = soil_result
            st.session_state.soil_location = selected_point
            st.session_state.location_soil_ph = soil_result.get("phh2o")

            climate_is_current = (
                not st.session_state.selected_climate.empty
                and st.session_state.climate_location == selected_point
            )
            if climate_is_current:
                st.session_state.location_suitability = build_location_suitability(
                    st.session_state.selected_climate,
                    ecological,
                    catalog,
                    soil_ph=soil_result.get("phh2o"),
                    soil_texture=soil_result,
                )
                st.session_state.location_rotations = build_location_rotations(
                    st.session_state.location_suitability
                )
            if soil_result.get("phh2o") is None:
                st.warning(
                    "SoilGrids returned texture data but no valid pH estimate for this coordinate. "
                    "Crop screening continues without the pH component."
                )
            elif soil_result.get("property_errors"):
                st.warning(
                    "Some SoilGrids properties were unavailable. Available properties are shown; "
                    "missing values are not imputed."
                )
            else:
                st.success(
                    "SoilGrids data loaded. Soil pH was added to crop screening "
                    "where climate data are available."
                )
            st.rerun()
        except Exception as exc:
            st.error(f"SoilGrids data could not be loaded: {exc}")

    if soil_is_current:
        soil_values = st.session_state.selected_soil
        st.success(
            f"Showing SoilGrids estimates for "
            f"{selected_point[0]:.4f}, {selected_point[1]:.4f}"
        )
        m1, m2, m3, m4 = st.columns(4)
        ph_value = soil_values.get("phh2o")
        m1.metric("Soil pH (H₂O)", fmt(ph_value, 2) if ph_value is not None else "Not available")
        for column, key, label in [(m2, "sand", "Sand"), (m3, "silt", "Silt"), (m4, "clay", "Clay")]:
            value = soil_values.get(key)
            column.metric(label, f"{fmt(value, 1)}%" if value is not None else "Not available")

        texture_values = {
            "sand": soil_values.get("sand"),
            "silt": soil_values.get("silt"),
            "clay": soil_values.get("clay"),
        }
        available_texture = {
            k: float(v) for k, v in texture_values.items()
            if v is not None and pd.notna(v) and 0 <= float(v) <= 100
        }
        texture_complete = bool(soil_values.get("texture_complete"))
        if available_texture:
            texture_df = pd.DataFrame({
                "Soil particle": [k.capitalize() for k in available_texture],
                "Percentage": list(available_texture.values()),
            })
            section("Topsoil texture composition · 0–5 cm" if texture_complete else "Available texture fractions · incomplete")
            fig = px.bar(
                texture_df,
                x="Percentage",
                y="Soil particle",
                orientation="h",
                text_auto=".1f",
                color="Soil particle",
                title="Estimated soil texture fractions (%)",
            )
            fig.update_layout(
                template="plotly_white", height=300, showlegend=False,
                xaxis=dict(title="Percentage", range=[0, 100], ticksuffix="%"),
                margin=dict(l=15, r=15, t=35, b=15),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "SoilGrids did not return valid topsoil texture values for this point. "
                "The zero or missing raster cells were rejected instead of being shown "
                "as 0%. Try a nearby point or reload the soil data."
            )
        if soil_values.get("texture_sum_pct") is not None:
            st.caption(
                f"Texture fractions total: {fmt(soil_values['texture_sum_pct'], 1)}%. "
                "Small differences from 100% can occur because these are independently modelled estimates."
            )
        else:
            st.caption("Some texture fractions are unavailable at this coordinate; missing values are not estimated or filled in.")
        if not texture_complete:
            st.warning("The texture response is incomplete. Do not interpret the available fractions as a complete 100% soil composition or use them for texture classification.")
        if soil_values.get("property_errors"):
            with st.expander("SoilGrids request diagnostics"):
                for prop_name, error_text in soil_values["property_errors"].items():
                    st.write(f"**{prop_name}:** {error_text}")
        if soil_values.get("sample_distance_m"):
            st.caption("Nearest valid SoilGrids pixel distance (m): " + ", ".join(
                f"{k}: {v if v is not None else 'N/A'}" for k, v in soil_values["sample_distance_m"].items()
            ))
        st.caption(
            "Source: ISRIC SoilGrids 2.0 WCS · 250 m grid · mean prediction · 0–5 cm. "
            "These are model estimates, not field measurements."
        )
    else:
        st.info(
            "No SoilGrids result has been loaded for this selected coordinate yet. "
            "Click the button above to retrieve the local topsoil estimates."
        )

    st.markdown(
        "[SoilGrids WCS documentation](https://docs.isric.org/globaldata/soilgrids/wcs.html)"
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
        for c in ["Temperature fit", "Annual rainfall fit", "Soil pH fit", "Soil texture fit", "Preliminary score"]:
            display[c] = display[c].map(lambda x: "Not available" if pd.isna(x) else fmt(x))
        st.dataframe(display, hide_index=True, use_container_width=True)
        if "Soil pH fit" in view.columns:
            zero_ph_rows = view[
                pd.to_numeric(view["Soil pH fit"], errors="coerce").eq(0)
            ]
            if not zero_ph_rows.empty:
                names = ", ".join(zero_ph_rows["Crop"].astype(str).tolist())
                st.warning(
                    f"Soil pH fit is 0 for: {names}. A zero here means the measured pH is "
                    "outside that crop's ECOCROP absolute pH range; it does not mean the API returned zero."
                )
        soil_is_current_for_crop = (
            st.session_state.selected_soil is not None
            and st.session_state.soil_location == (
                st.session_state.selected_lat, st.session_state.selected_lon
            )
        )
        if soil_is_current_for_crop:
            current_soil = st.session_state.selected_soil or {}
            has_ph = current_soil.get("phh2o") is not None
            has_texture = bool(current_soil.get("texture_complete"))
            if has_ph and has_texture:
                st.success(
                    "SoilGrids pH and complete texture fractions are available for this exact point "
                    "and are included where the crop has corresponding ECOCROP requirements."
                )
            elif has_ph:
                st.warning(
                    f"SoilGrids pH ({fmt(current_soil.get('phh2o'))}) is included, but complete "
                    "sand/silt/clay data are unavailable. Results are partial-soil screening; "
                    "texture has not been imputed."
                )
            elif has_texture:
                st.warning(
                    "Complete SoilGrids texture is included, but pH is unavailable. "
                    "Results are partial-soil screening."
                )
            else:
                st.warning(
                    "SoilGrids returned no complete soil component for this point. "
                    "Scores use climate data only; missing soil values are not treated as zero."
                )
        else:
            st.warning(
                "These are climate-only preliminary screening scores. "
                "Load SoilGrids data for this same coordinate to include available soil data."
            )
        st.caption(
            "Scores are preliminary screening indicators, not yield predictions or agronomic prescriptions. "
            "Weights when available: temperature 30%, annual rainfall 25%, soil pH 20%, and soil texture 25%; "
            "missing components are excluded and weights are renormalized. Texture is mapped to broad "
            "light/medium/heavy ECOCROP descriptors. Annual indicators do not replace crop-season or "
            "field-level assessment."
        )
    else:
        st.info(
            "No current location-specific screening is available yet. "
            "Go to Overview, select your point, and load NASA POWER climate data. "
            "The original Antalya pilot scores are intentionally hidden here to avoid "
            "presenting them as results for another location."
        )
        st.caption(
            f"Current selected point: {st.session_state.selected_lat:.4f}, "
            f"{st.session_state.selected_lon:.4f}"
        )

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
            "Rotation scenarios are heuristic candidates based on the available climate and soil screening data. "
            "They do not account for field history, planting calendars, pests, nutrient balances, "
            "irrigation constraints, or farm economics. They are not agronomic prescriptions."
        )
    else:
        if location_current and not st.session_state.location_suitability.empty:
            st.warning(
                "No valid three-year rotation can be generated from the crops that pass the current "
                "eligibility screen. This is intentional: SmartCrop will not fill a rotation with "
                "crops outside their absolute climate or soil-pH limits. Try another location or review "
                "the crop screening inputs."
            )
        else:
            st.info(
                "Select a point on the Overview map and load its NASA POWER climate data to generate "
                "location-specific rotation candidates."
            )
        st.caption(
            "Only eligible annual crops are considered. Perennial crops such as olive are excluded. "
            "Each crop appears once and all three years must belong to distinct crop groups."
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
        "Selected-point climate data are fetched from the NASA POWER Monthly Point API for the 1991–2020 reference period. This is a historical 30-year baseline, not a forecast.",
        "NASA POWER grid resolution varies by parameter; nearby selected points may return the same grid-cell data.",
        "Global point selection fetches NASA POWER climate data and SoilGrids topsoil estimates. When available for the same point, crop screening includes temperature, annual rainfall, soil pH, and broad soil-texture compatibility; crops without an ECOCROP texture requirement are not penalized for missing texture metadata.",
        "SoilGrids 0–5 cm pH and texture values are gridded model estimates and have not been validated with field soil samples.",
        "The Antalya pilot pH value is a nearest-valid-pixel estimate. It is not reused for other selected locations.",
        "Rainfall fit compares the mean annual precipitation total derived from 1991–2020 monthly climate records with ECOCROP reference ranges. It is not irrigation demand.",
        "For a selected location, the preliminary score combines available temperature (30%), annual rainfall (25%), soil pH (20%), and soil texture (25%) fits; missing components are excluded and weights are renormalized. Temperature is a day-weighted annual mean of monthly normals. Texture is classified approximately using the USDA textural triangle and mapped to broad ECOCROP light/medium/heavy descriptors; it is not a laboratory soil classification.",
        "Root-zone water storage is estimated using generalized rooting-depth references and SoilGrids layer values.",
        "Root-zone storage is not plant water uptake at a particular date and is not an irrigation prescription.",
        "The rotation generator uses only crops passing the current screening eligibility and requires three distinct crop groups in each three-year sequence. Perennial crops such as olive are excluded. It does not model field history, planting calendars, pests, diseases, nutrient balances, soil carbon, irrigation, or economics.",
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
