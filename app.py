import html
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
    initial_sidebar_state="collapsed",
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
    """Fetch 0–5 cm SoilGrids mean estimates via WCS, with bounded retries for transient service timeouts.

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
        """Fetch one WCS layer with bounded retries for transient ISRIC timeouts."""
        import time
        endpoint = f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
        params = [
            ("SERVICE", "WCS"), ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"), ("COVERAGEID", f"{prop}_0-5cm_mean"),
            ("FORMAT", "GEOTIFF_INT16"), *subsets,
            ("SUBSETTINGCRS", crs_uri), ("OUTPUTCRS", crs_uri),
            ("RESX", "250"), ("RESY", "250"),
        ]
        last_error = None
        # ISRIC occasionally stalls on individual WCS layers. Retry only transient
        # failures; never replace missing measurements with fabricated values.
        for attempt in range(3):
            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    headers={
                        "User-Agent": "SmartCrop/1.3 (educational research client)",
                        "Accept": "image/tiff, image/*;q=0.9, */*;q=0.8",
                    },
                    timeout=(15, 100),
                )
                response.raise_for_status()
                content = response.content
                if not content:
                    raise RuntimeError("SoilGrids returned an empty response.")

                # Do not reject by Content-Type: valid GeoTIFFs may be labelled image/tiff.
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
                                valid &= (values >= 0) & (values <= 1000)
                            return values.copy(), valid.copy(), ds.transform, ds.crs
                except Exception as exc:
                    content_type = response.headers.get("Content-Type", "unknown")
                    raise RuntimeError(
                        f"Could not open WCS response as GeoTIFF (HTTP {response.status_code}, "
                        f"Content-Type: {content_type}): {exc}"
                    ) from exc
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}")

    # Limit concurrency to avoid overwhelming the public WCS service; each layer
    # has independent retries so one failed property does not discard other layers.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=2) as executor:
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
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {display:none !important;}
[data-testid="stMainBlockContainer"] {padding-top:184px !important;}
.st-key-topnav {
  background:#10382d;
  position:fixed;
  top:0;
  left:0;
  width:100vw;
  max-width:100vw;
  margin:0;
  border-radius:0;
  box-sizing:border-box;
  padding:0 32px;
  height:168px;
  min-height:168px;
  z-index:1000001;
  overflow:visible;
  display:flex;
  align-items:center;
}
.st-key-topnav [data-testid="stVerticalBlock"],
.st-key-topnav [data-testid="stHorizontalBlock"],
.st-key-topnav [data-testid="stElementContainer"],
.st-key-topnav [data-testid="stColumn"] {overflow:visible !important;}
.st-key-topnav [data-testid="stVerticalBlock"] {
  width:100%;
  justify-content:center !important;
}
.st-key-topnav [data-testid="stHorizontalBlock"] {
  align-items:center !important;
  width:100%;
}
.st-key-topnav [data-testid="stColumn"] {
  display:flex !important;
  align-items:center !important;
}
.st-key-topnav [data-testid="stColumn"] [data-testid="stVerticalBlock"] {
  justify-content:center !important;
}
.st-key-topnav [data-testid="stHorizontalBlock"],
.st-key-topnav [data-testid="stColumn"] {
  height:168px !important;
  min-height:168px !important;
  align-items:center !important;
}
.st-key-topnav [data-testid="stColumn"]:nth-child(2),
.st-key-topnav [data-testid="stColumn"]:nth-child(2) [data-testid="stVerticalBlock"],
.st-key-topnav [data-testid="stColumn"]:nth-child(2) [data-testid="stElementContainer"],
.st-key-topnav [data-testid="stRadio"],
.st-key-topnav [data-testid="stRadio"] > div,
.st-key-topnav [role="radiogroup"] {
  height:168px !important;
  min-height:168px !important;
  width:100% !important;
  max-width:100% !important;
  align-items:stretch !important;
}
.st-key-topnav [data-testid="stRadio"] > div[role="radiogroup"] {
  display:flex !important;
  justify-content:space-between !important;
  align-items:center !important;
  gap:8px;
}
.st-key-topnav [data-testid="stWidgetLabel"] {display:none !important;min-height:0 !important;padding:0 !important;margin:0 !important;}
.st-key-topnav label[data-baseweb="radio"] > div:first-child,
.st-key-topnav label[data-baseweb="radio"] input {
  display:none !important;
  width:0 !important;
  height:0 !important;
  margin:0 !important;
  padding:0 !important;
}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] label,
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] label p,
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] label span {color:#f4f8f4 !important;font-weight:600;line-height:1.35 !important;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] label {
  height:52px !important;
  min-height:52px !important;
  margin:0 !important;
  padding:0 14px !important;
  border-radius:12px !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  white-space:nowrap;
  background:transparent;
  cursor:pointer;
}
.st-key-topnav [data-testid="stRadio"] label:hover,
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) {background:#f4f8f4;}
.st-key-topnav [data-testid="stRadio"] label:hover p,
.st-key-topnav [data-testid="stRadio"] label:hover span,
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) p,
.st-key-topnav [data-testid="stRadio"] label:has(input:checked) span {color:#10382d !important;background:transparent !important;}
.st-key-open_location button {
  background:rgba(255,255,255,.08);
  color:#f4f8f4;
  border:1px solid rgba(255,255,255,.22);
  border-radius:14px;
  min-height:68px;
  white-space:normal;
  line-height:1.3;
  text-align:left;
  padding:8px 12px;
}
.st-key-open_location button p {color:#f4f8f4 !important;font-weight:700;}
.st-key-rail_soil, .st-key-rail_climate {
  position:fixed;
  top:168px;
  bottom:0;
  width:72px;
  z-index:1000000;
}
.st-key-rail_soil {right:0;}
.st-key-rail_climate {left:0;}
.st-key-rail_soil button, .st-key-rail_climate button {
  width:72px;
  height:calc(100vh - 168px);
  min-height:calc(100vh - 168px);
  background:#10382d;
  color:#f4f8f4;
  border:0;
  border-radius:0;
  font-weight:700;
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:visible !important;
  position:relative;
}
.st-key-rail_soil button [data-testid="stMarkdownContainer"],
.st-key-rail_climate button [data-testid="stMarkdownContainer"] {
  position:relative;
  width:max-content !important;
  max-width:none !important;
  overflow:visible !important;
  transform:rotate(90deg);
  transform-origin:center center;
}
.st-key-rail_climate button [data-testid="stMarkdownContainer"]::after,
.st-key-rail_soil button [data-testid="stMarkdownContainer"]::after {
  position:absolute;
  left:calc(100% + 12px);
  top:50%;
  transform:translateY(-50%) rotate(-90deg);
  font-size:22px;
  line-height:1;
  color:#f4f8f4;
  pointer-events:none;
}
.st-key-rail_climate button [data-testid="stMarkdownContainer"]::after {content:"←";}
.st-key-rail_soil button [data-testid="stMarkdownContainer"]::after {content:"→";}
.st-key-rail_climate button:hover [data-testid="stMarkdownContainer"]::after,
.st-key-rail_soil button:hover [data-testid="stMarkdownContainer"]::after,
.st-key-rail_climate button:focus [data-testid="stMarkdownContainer"]::after,
.st-key-rail_soil button:focus [data-testid="stMarkdownContainer"]::after {color:#10382d;}
.st-key-rail_soil button p, .st-key-rail_climate button p {
  color:#f4f8f4 !important;
  writing-mode:horizontal-tb !important;
  text-orientation:mixed !important;
  white-space:nowrap !important;
  width:max-content !important;
  max-width:none !important;
  height:auto !important;
  margin:0 !important;
  line-height:1;
  text-align:center;
  letter-spacing:.4px;
  font-size:16px;
  font-weight:700;
}
.st-key-rail_soil button:hover p,
.st-key-rail_climate button:hover p,
.st-key-rail_soil button:focus p,
.st-key-rail_climate button:focus p,
.st-key-rail_soil button:hover div,
.st-key-rail_climate button:hover div,
.st-key-rail_soil button:focus div,
.st-key-rail_climate button:focus div {
  color:#10382d !important;
}
.st-key-mobile_menu_btn {display:none !important;}
@media (max-width: 820px) {
  [data-testid="stMainBlockContainer"] {padding-top:82px !important;}
  .st-key-mobile_menu_btn {
    display:block !important;
    position:absolute !important;
    top:10px;
    right:12px;
    width:44px !important;
    height:44px !important;
    min-height:0 !important;
    margin:0 !important;
    z-index:5;
  }
  .st-key-mobile_menu_btn button {
    width:44px;
    height:44px;
    min-height:44px;
    padding:0;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.22);
    border-radius:12px;
  }
  .st-key-mobile_menu_btn button p {color:#f4f8f4 !important;font-size:22px;line-height:1;}
  .st-key-topnav [data-testid="stColumn"]:nth-child(1) {
    min-height:48px !important;
    padding-right:56px;
    box-sizing:border-box;
  }
  .st-key-topnav [data-testid="stColumn"]:nth-child(2) {display:none !important;}
  .st-key-topnav:has(#mobile-menu-open) [data-testid="stColumn"]:nth-child(2) {
    display:flex !important;
    width:100% !important;
  }
  .st-key-topnav:has(#mobile-menu-open) [data-testid="stRadio"] > div[role="radiogroup"] {
    flex-direction:column !important;
    align-items:stretch !important;
    overflow:visible;
  }
  .st-key-topnav:has(#mobile-menu-open) [data-testid="stRadio"] [role="radiogroup"] label {
    width:100%;
    justify-content:flex-start !important;
  }
  .st-key-topnav {
    height:auto;
    min-height:0;
    padding:8px 12px 10px;
    align-items:stretch;
  }
  .st-key-topnav [data-testid="stHorizontalBlock"],
  .st-key-topnav [data-testid="stColumn"],
  .st-key-topnav [data-testid="stColumn"]:nth-child(2),
  .st-key-topnav [data-testid="stColumn"]:nth-child(2) [data-testid="stVerticalBlock"],
  .st-key-topnav [data-testid="stColumn"]:nth-child(2) [data-testid="stElementContainer"],
  .st-key-topnav [data-testid="stRadio"],
  .st-key-topnav [data-testid="stRadio"] > div,
  .st-key-topnav [role="radiogroup"] {
    height:auto !important;
    min-height:0 !important;
  }
  .st-key-topnav [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;row-gap:6px;}
  .st-key-topnav [data-testid="stColumn"] {
    width:100% !important;
    flex:1 1 100% !important;
    min-width:0 !important;
  }
  .st-key-topnav [data-testid="stRadio"] > div[role="radiogroup"] {
    justify-content:flex-start !important;
    overflow-x:auto;
    padding-bottom:2px;
  }
  .st-key-topnav [data-testid="stRadio"] [role="radiogroup"] label {
    height:40px !important;
    min-height:40px !important;
    padding:0 12px !important;
  }
  .st-key-open_location button {min-height:44px;width:100%;}
  .hero {padding:18px 16px;}
  .hero h1 {font-size:1.55rem;}
  .st-key-rail_soil, .st-key-rail_climate {
    top:auto;
    bottom:0;
    height:64px;
    width:50%;
    overflow:hidden;
  }
  .st-key-rail_climate {left:0;right:auto;}
  .st-key-rail_soil {right:0;left:auto;box-shadow:inset 1px 0 0 rgba(255,255,255,.18);}
  body:not(:has(.st-key-rail_soil)) .st-key-rail_climate,
  body:not(:has(.st-key-rail_climate)) .st-key-rail_soil {
    width:100%;
    left:0;
    right:0;
    box-shadow:none;
  }
  .st-key-rail_soil button, .st-key-rail_climate button {
    width:100%;
    height:64px;
    min-height:64px;
    overflow:hidden !important;
  }
  .st-key-rail_soil button [data-testid="stMarkdownContainer"],
  .st-key-rail_climate button [data-testid="stMarkdownContainer"] {
    transform:none !important;
    width:auto !important;
    max-width:100% !important;
  }
  .st-key-rail_climate button [data-testid="stMarkdownContainer"]::after,
  .st-key-rail_soil button [data-testid="stMarkdownContainer"]::after {
    content:none !important;
    display:none !important;
  }
  .st-key-rail_climate button p::before {content:"←  ";}
  .st-key-rail_soil button p::after {content:"  →";}
  .st-key-rail_soil button p, .st-key-rail_climate button p {
    font-size:15px;
    letter-spacing:0;
  }
}
.brand {display:flex;align-items:center;gap:8px;color:#f4f8f4;padding-top:0;}
.brand strong {font-family:Manrope,sans-serif;font-size:1.15rem;display:block;line-height:1.1;}
.brand small {color:#b7d7c4;font-size:.72rem;letter-spacing:.4px;}
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
.facts {display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:8px 0 18px;}
.fact {background:#fff;border:1px solid #e0e9e1;border-radius:14px;padding:12px 14px;}
.fact span {display:block;color:#52665b;font-size:.78rem;font-weight:600;}
.fact strong {display:block;color:#173f2e;font-size:1rem;margin-top:3px;}
[data-testid="stCustomComponentV1"] {position:relative;z-index:0;}
div[data-baseweb="popover"], div[data-baseweb="menu"] {z-index:1000002 !important;}
.location-chip {background:rgba(255,255,255,.08);color:#f4f8f4;border-radius:14px;padding:4px 12px 6px;min-width:160px;}
.location-chip-kicker {font-size:10px;letter-spacing:1.4px;font-weight:700;color:#b7d7c4;}
.location-chip-name {font-family:Manrope,sans-serif;font-weight:800;font-size:.88rem;line-height:1.2;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.location-chip-meta {font-size:.72rem;color:#d5e8dc;margin-top:0;}
</style>
""", unsafe_allow_html=True)


def _http_status(exc):
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def explain_climate_failure(exc):
    """Separate a NASA coverage gap from a temporary service failure."""
    status = _http_status(exc)
    if (
        isinstance(exc, (requests.Timeout, requests.ConnectionError))
        or (status is not None and (status >= 500 or status == 429))
    ):
        return (
            "service",
            "NASA POWER did not respond. This is a temporary service problem, "
            "not a system error in SmartCrop, and it does not mean this point has no climate record. "
            "Wait a moment and try again.",
        )
    if isinstance(exc, ValueError):
        return (
            "unavailable",
            "NASA POWER has no usable 1991–2020 climate record for this coordinate. "
            "This is not a system error. NASA did not return enough data for this point. "
            "Choose another point on land and load the climate record again.",
        )
    return (
        "unexpected",
        "The climate request failed for an unexpected reason. This point was not scored.",
    )


def explain_soil_failure(exc):
    """Separate a SoilGrids coverage gap from a temporary service failure."""
    lowered = str(exc).lower()
    if "dependencies are missing" in lowered:
        return (
            "unavailable",
            "SoilGrids cannot be read because a required library is missing. "
            "Climate screening is unaffected.",
        )
    status = _http_status(exc)
    service_tokens = ("timeout", "timed out", "connection", "502", "503", "504", "429")
    if (
        isinstance(exc, (requests.Timeout, requests.ConnectionError))
        or (status is not None and (status >= 500 or status == 429))
        or any(token in lowered for token in service_tokens)
    ):
        return (
            "service",
            "ISRIC SoilGrids did not respond. This is a temporary problem with the soil service, "
            "not a NASA climate gap and not a system error in SmartCrop. "
            "Try again shortly. Crop screening can continue with climate data alone.",
        )
    if isinstance(exc, ValueError):
        return (
            "unavailable",
            "SoilGrids has no usable 0–5 cm estimate for this coordinate. "
            "This is not a system error. The soil map often has no value over water "
            "or on some coastal pixels. Choose a nearby point on land.",
        )
    return (
        "unexpected",
        "The soil request failed for an unexpected reason. Climate screening is unchanged.",
    )


def notice_from(exc, explainer):
    kind, text = explainer(exc)
    return {"kind": kind, "text": text, "detail": f"{type(exc).__name__}: {exc}"}


def show_remote_notice(notice):
    if not notice:
        return
    if notice.get("kind") == "unexpected":
        st.error(notice.get("text", ""))
    else:
        st.warning(notice.get("text", ""))
    detail = notice.get("detail")
    if detail:
        with st.expander("Technical detail"):
            st.code(detail)


def current_point():
    return (st.session_state.selected_lat, st.session_state.selected_lon)


def climate_is_current():
    return (
        st.session_state.location_chosen
        and not st.session_state.selected_climate.empty
        and st.session_state.climate_location == current_point()
    )


def soil_is_current():
    return (
        st.session_state.location_chosen
        and st.session_state.selected_soil is not None
        and st.session_state.soil_location == current_point()
    )


def clear_point_results():
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
    st.session_state.climate_notice = None
    st.session_state.soil_notice = None


def render_selected_place():
    """Place button at the right of the top menu. Opens the location dialog."""
    if st.session_state.location_chosen:
        name = st.session_state.location_label or "Map point"
        meta = f"{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}"
        label = f"{name} · {meta}"
    else:
        label = "Choose a place · Country, city, or map"
    return st.button(label, key="open_location", width="stretch")


def render_topnav():
    """Fixed page menu. The place button on the right reopens location selection."""
    pages = [
        "Location",
        "Climate Analysis",
        "Soil Analysis",
        "Crop Suitability",
        "Crop Rotation",
        "Data Sources",
    ]
    with st.container(key="topnav"):
        menu_open = bool(st.session_state.get("mobile_menu_open"))
        if st.button("✕" if menu_open else "☰", key="mobile_menu_btn", help="Menu"):
            st.session_state.mobile_menu_open = not menu_open
            st.rerun()
        if st.session_state.get("mobile_menu_open"):
            st.markdown('<div id="mobile-menu-open"></div>', unsafe_allow_html=True)
        brand, nav, place = st.columns([1.05, 5.4, 2.15], vertical_alignment="center")
        with brand:
            st.markdown(
                '<div class="brand"><span>🌱</span><div>'
                "<strong>SmartCrop</strong><small>Field intelligence</small></div></div>",
                unsafe_allow_html=True,
            )
        with nav:
            page = st.radio(
                "Pages",
                pages,
                horizontal=True,
                label_visibility="collapsed",
                key="nav_page",
                format_func=lambda name: {
                    "Climate Analysis": "Climate",
                    "Soil Analysis": "Soil",
                    "Crop Suitability": "Suitability",
                    "Crop Rotation": "Rotation",
                    "Data Sources": "Sources",
                }.get(name, name),
            )
            if st.session_state.get("_menu_for_page") not in (None, page):
                st.session_state.mobile_menu_open = False
                st.session_state._menu_for_page = page
                st.rerun()
            st.session_state._menu_for_page = page
        with place:
            reopen_location = render_selected_place()
    return page, reopen_location


PLACE_UNSET = "Select"
PLACES_API = "https://countriesnow.space/api/v0.1"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PLACE_HEADERS = {"User-Agent": "SmartCrop/1.3 (educational research client)"}


def explain_place_failure(exc):
    """Separate a place-directory outage from a name that has no coordinates."""
    status = _http_status(exc)
    if (
        isinstance(exc, (requests.Timeout, requests.ConnectionError))
        or (status is not None and (status >= 500 or status == 429))
    ):
        return (
            "service",
            "The place list did not respond. This is a temporary service problem, "
            "not a system error in SmartCrop. Refresh the page to try again, or click the map.",
        )
    return (
        "unavailable",
        "The place list could not provide that entry. This is not a system error. "
        "Choose another place or click the map.",
    )


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_place_countries():
    response = requests.get(f"{PLACES_API}/countries/positions", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("msg") or "The place directory returned an error.")
    rows = []
    for item in payload.get("data") or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            lat, lon = float(item.get("lat")), float(item.get("long"))
        except (TypeError, ValueError):
            lat, lon = None, None
        rows.append({"name": name, "lat": lat, "lon": lon})
    rows.sort(key=lambda row: row["name"].casefold())
    return rows


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_place_regions(country):
    response = requests.post(
        f"{PLACES_API}/countries/states",
        json={"country": country},
        timeout=30,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        return []
    choices, seen = [], set()
    for state in (payload.get("data") or {}).get("states") or []:
        api_name = str(state.get("name") or "").strip()
        if not api_name:
            continue
        label = api_name[:-9] if api_name.endswith(" Province") else api_name
        if label.casefold() in seen:
            label = api_name
        seen.add(label.casefold())
        choices.append({"label": label, "value": api_name})
    choices.sort(key=lambda row: row["label"].casefold())
    return choices


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_place_districts(country, region):
    response = requests.post(
        f"{PLACES_API}/countries/state/cities",
        json={"country": country, "state": region},
        timeout=30,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        return []
    grouped = {}
    for name in payload.get("data") or []:
        raw = str(name).strip()
        if not raw:
            continue
        label = raw.replace(" İlçesi", "").replace(" Ilçesi", "").strip()
        grouped.setdefault(label.casefold(), []).append(raw)
    choices = []
    for names in grouped.values():
        preferred = next((name for name in names if "İlçesi" in name or "Ilçesi" in name), names[0])
        label = preferred.replace(" İlçesi", "").replace(" Ilçesi", "").strip()
        choices.append({"label": label, "value": preferred})
    choices.sort(key=lambda row: row["label"].casefold())
    return choices


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_place(query, expected_name):
    """Return coordinates for the result whose name matches the chosen place."""
    response = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "limit": 5},
        headers=PLACE_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    expected = "".join(ch for ch in expected_name.casefold() if ch.isalnum())
    if not expected:
        return None
    for item in payload or []:
        head = str(item.get("display_name") or "").split(",")[0]
        actual = "".join(ch for ch in head.casefold() if ch.isalnum())
        if actual == expected or actual.startswith(expected) or expected.startswith(actual):
            return float(item["lat"]), float(item["lon"])
    return None


def adopt_point(lat, lon, label):
    """Store a pin. Clear loaded climate and soil only when the point actually moves."""
    moved = (
        abs(float(lat) - st.session_state.selected_lat) > 0.0001
        or abs(float(lon) - st.session_state.selected_lon) > 0.0001
    )
    first = not st.session_state.location_chosen
    st.session_state.location_chosen = True
    st.session_state.selected_lat = float(lat)
    st.session_state.selected_lon = float(lon)
    st.session_state.location_label = label
    if moved:
        clear_point_results()
    return first or moved


def begin_point_load():
    """A new pin loads climate and soil on its own, then opens Climate."""
    st.session_state.pending_point_load = True
    st.rerun()


def render_place_selectors(prefix=""):
    """Country, city, and district lists. A completed choice moves the pin."""
    country_key = f"{prefix}place_country"
    region_key = f"{prefix}place_region"
    district_key = f"{prefix}place_district"
    seen_country_key = f"{prefix}place_country_seen"
    seen_region_key = f"{prefix}place_region_seen"
    section("Choose a country, city, and district")
    st.caption(
        "City is the province or state. District is a locality inside it. "
        "You can also click the map below. Data loads as soon as the pin is set."
    )
    try:
        countries = fetch_place_countries()
    except Exception as exc:
        show_remote_notice(notice_from(exc, explain_place_failure))
        return None

    if st.session_state.pop(f"{prefix}reset_place_lists", False):
        st.session_state[country_key] = PLACE_UNSET
        st.session_state[region_key] = PLACE_UNSET
        st.session_state[district_key] = PLACE_UNSET
        st.session_state[seen_country_key] = PLACE_UNSET
        st.session_state[seen_region_key] = (PLACE_UNSET, PLACE_UNSET)
        st.session_state.applied_place_key = None
        st.session_state.place_notice = None

    by_name = {row["name"]: row for row in countries}
    country_col, city_col, district_col = st.columns(3)
    with country_col:
        country = st.selectbox("Country", [PLACE_UNSET] + list(by_name), key=country_key)
    if st.session_state.get(seen_country_key) != country:
        st.session_state[region_key] = PLACE_UNSET
        st.session_state[district_key] = PLACE_UNSET
        st.session_state[seen_country_key] = country

    regions, region_error = [], None
    if country != PLACE_UNSET:
        try:
            regions = fetch_place_regions(country)
        except Exception as exc:
            region_error = notice_from(exc, explain_place_failure)
    region_by_label = {row["label"]: row["value"] for row in regions}
    region_options = [PLACE_UNSET] + list(region_by_label)
    if st.session_state.get(region_key, PLACE_UNSET) not in region_options:
        st.session_state[region_key] = PLACE_UNSET
    with city_col:
        region_label = st.selectbox(
            "City",
            region_options,
            key=region_key,
            disabled=country == PLACE_UNSET,
        )
    if st.session_state.get(seen_region_key) != (country, region_label):
        st.session_state[district_key] = PLACE_UNSET
        st.session_state[seen_region_key] = (country, region_label)

    region_value = region_by_label.get(region_label)
    districts, district_error = [], None
    if region_value:
        try:
            districts = fetch_place_districts(country, region_value)
        except Exception as exc:
            district_error = notice_from(exc, explain_place_failure)
    district_by_label = {row["label"]: row["value"] for row in districts}
    district_options = [PLACE_UNSET] + list(district_by_label)
    if st.session_state.get(district_key, PLACE_UNSET) not in district_options:
        st.session_state[district_key] = PLACE_UNSET
    with district_col:
        district_label = st.selectbox(
            "District",
            district_options,
            key=district_key,
            disabled=not districts,
        )
    show_remote_notice(region_error or district_error)

    district_value = district_by_label.get(district_label)
    place_key, queries, place_label = None, [], None
    if region_value and district_value:
        place_key = (country, region_value, district_value)
        place_label = f"{district_label}, {region_label}, {country}"
        queries = [
            f"{district_label}, {region_label}, {country}",
            f"{district_value}, {region_value}, {country}",
        ]
    elif region_value and not districts and district_error is None:
        place_key = (country, region_value, "")
        place_label = f"{region_label}, {country}"
        queries = [f"{region_label}, {country}", f"{region_value}, {country}"]
        st.caption("This city has no district list. The pin uses the city center.")

    if place_key and place_key != st.session_state.applied_place_key:
        try:
            found = None
            target_name = district_label if district_value else region_label
            for query in queries:
                found = geocode_place(query, target_name)
                if found:
                    break
            st.session_state.applied_place_key = place_key
            if not found:
                st.session_state.place_notice = {
                    "kind": "unavailable",
                    "text": (
                        "No coordinates were found for this place. This is not a system error. "
                        "Choose another district or click the map."
                    ),
                }
            else:
                st.session_state.place_notice = None
                if adopt_point(found[0], found[1], place_label):
                    begin_point_load()
        except Exception as exc:
            st.session_state.applied_place_key = place_key
            st.session_state.place_notice = notice_from(exc, explain_place_failure)
    show_remote_notice(st.session_state.place_notice)

    if not st.session_state.location_chosen and country != PLACE_UNSET:
        row = by_name.get(country) or {}
        if row.get("lat") is not None and row.get("lon") is not None:
            return (row["lat"], row["lon"], 4)
    return None


def render_location_map(prefix=""):
    """Full-width map. A pin exists only after the user chooses a point."""
    preview = render_place_selectors(prefix)
    section("Or click the map")
    st.caption(
        "A map click places the pin directly. "
        "Climate and soil data load on their own, and the Climate page opens."
    )
    center_lat = st.session_state.selected_lat
    center_lon = st.session_state.selected_lon
    zoom = 5 if st.session_state.location_chosen else 2
    if preview and not st.session_state.location_chosen:
        center_lat, center_lon, zoom = preview
    map_view = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
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
    if st.session_state.location_chosen:
        folium.Marker(
            [st.session_state.selected_lat, st.session_state.selected_lon],
            tooltip="Selected SmartCrop location",
            popup=(
                f"Selected point: {st.session_state.selected_lat:.4f}, "
                f"{st.session_state.selected_lon:.4f}"
            ),
            icon=folium.Icon(color="green"),
        ).add_to(map_view)
    map_result = st_folium(
        map_view,
        height=520,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key=f"{prefix}location_map",
    )
    clicked = map_result.get("last_clicked") if map_result else None
    if clicked:
        new_lat = float(clicked["lat"])
        new_lon = float(clicked["lng"])
        moved = (
            abs(new_lat - st.session_state.selected_lat) > 0.0001
            or abs(new_lon - st.session_state.selected_lon) > 0.0001
        )
        if not st.session_state.location_chosen or moved:
            st.session_state[f"{prefix}reset_place_lists"] = True
            if adopt_point(new_lat, new_lon, "Chosen on the map"):
                begin_point_load()


def prompt_for_location():
    st.warning("No location is selected. Choose a place on the Location page before reading this report.")
    if st.button("Go to Location", key="go_to_location", type="primary"):
        st.session_state.pending_nav = "Location"
        st.rerun()


def render_report_rail(side, label, target):
    """Full-height green rail. Right continues the report, left returns."""
    st.markdown(
        """
        <style>
        @media (min-width: 821px) {
          [data-testid='stMainBlockContainer']{padding-left:96px !important;padding-right:96px !important;}
        }
        @media (max-width: 820px) {
          [data-testid='stMainBlockContainer']{padding-left:12px !important;padding-right:12px !important;padding-bottom:84px !important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    key = "rail_soil" if side == "right" else "rail_climate"
    if st.button(label, key=key):
        st.session_state.pending_nav = target
        st.rerun()


def prompt_for_climate():
    st.info(
        "Climate data loads automatically when a place is chosen. "
        "This page stays empty until NASA POWER returns a 1991–2020 record for that pin."
    )
    show_remote_notice(st.session_state.climate_notice)


def load_selected_point(ecological, catalog):
    """Fetch climate and soil for the current pin. No extra button."""
    lat = st.session_state.selected_lat
    lon = st.session_state.selected_lon
    try:
        climate_package = fetch_nasa_power_climate_normals(lat, lon, 1991, 2020)
        result = climate_package["monthly_normals"]
        st.session_state.selected_climate = result
        st.session_state.selected_climate_history = climate_package["monthly_history"]
        st.session_state.selected_annual_climate = climate_package["annual_summary"]
        st.session_state.climate_period = climate_package["period"]
        st.session_state.climate_location = current_point()
        st.session_state.climate_notice = None
    except Exception as exc:
        st.session_state.climate_notice = notice_from(exc, explain_climate_failure)
        st.session_state.selected_climate = pd.DataFrame()
        st.session_state.selected_climate_history = pd.DataFrame()
        st.session_state.selected_annual_climate = pd.DataFrame()
        st.session_state.climate_period = None
        st.session_state.climate_location = None
        st.session_state.location_suitability = pd.DataFrame()
        st.session_state.location_rotations = pd.DataFrame()

    try:
        soil_result = fetch_soilgrids_point(lat, lon)
        st.session_state.selected_soil = soil_result
        st.session_state.soil_location = current_point()
        st.session_state.location_soil_ph = soil_result.get("phh2o")
        ph_missing = soil_result.get("phh2o") is None
        texture_ok = bool(soil_result.get("texture_complete"))
        if ph_missing and not texture_ok:
            notice_text = (
                "SoilGrids has no usable pH or complete texture for this pin. "
                "This is a data gap, not a system error. Missing values were not filled in."
            )
        elif ph_missing:
            notice_text = (
                "SoilGrids returned texture but no valid pH for this pin. "
                "Screening continues without pH. This is a data gap, not a system error."
            )
        elif not texture_ok:
            notice_text = (
                "SoilGrids pH is available, but the texture fractions are incomplete. "
                "Texture was not guessed or filled in."
            )
        elif soil_result.get("property_errors"):
            notice_text = (
                "Some SoilGrids properties were unavailable. "
                "Available values are shown and missing values are not filled in."
            )
        else:
            notice_text = None
        st.session_state.soil_notice = (
            {"kind": "unavailable", "text": notice_text} if notice_text else None
        )
    except Exception as exc:
        st.session_state.soil_notice = notice_from(exc, explain_soil_failure)
        st.session_state.selected_soil = None
        st.session_state.soil_location = None
        st.session_state.location_soil_ph = None

    if climate_is_current():
        matching_soil = st.session_state.selected_soil if soil_is_current() else None
        selected_ph = matching_soil.get("phh2o") if matching_soil else None
        st.session_state.location_soil_ph = selected_ph
        st.session_state.location_suitability = build_location_suitability(
            st.session_state.selected_climate,
            ecological,
            catalog,
            soil_ph=selected_ph,
            soil_texture=matching_soil,
        )
        st.session_state.location_rotations = build_location_rotations(
            st.session_state.location_suitability
        )


def render_point_facts():
    """Listed technical values for the selected pin."""
    section("Technical record")
    lat = st.session_state.selected_lat
    lon = st.session_state.selected_lon
    rows = [
        ("Place", st.session_state.location_label or "Map point"),
        ("Coordinates", f"{lat:.4f}, {lon:.4f}"),
    ]
    if climate_is_current():
        frame = st.session_state.selected_climate
        temperature = pd.to_numeric(frame.get("avg_temperature"), errors="coerce")
        rainfall = pd.to_numeric(frame.get("total_rainfall"), errors="coerce")
        rows.append(("Climate period", str(st.session_state.climate_period or "1991–2020")))
        rows.append(("Mean monthly temperature", f"{fmt(temperature.mean())} °C"))
        rows.append(("Mean annual precipitation", f"{fmt(rainfall.sum(), 0)} mm"))
        if "avg_solar_radiation" in frame.columns:
            solar = pd.to_numeric(frame["avg_solar_radiation"], errors="coerce")
            rows.append(("Mean daily solar radiation", f"{fmt(solar.mean())} MJ/m²/day"))
    else:
        rows.append(("Climate", "Not available"))
    if soil_is_current():
        soil_values = st.session_state.selected_soil or {}
        ph_value = soil_values.get("phh2o")
        rows.append(("Soil pH", fmt(ph_value, 2) if ph_value is not None else "Not available"))
        for key, label in (("sand", "Sand"), ("silt", "Silt"), ("clay", "Clay")):
            value = soil_values.get(key)
            rows.append((label, f"{fmt(value, 1)}%" if value is not None else "Not available"))
    else:
        rows.append(("Soil", "Not available"))
    items = "".join(
        f'<div class="fact"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(str(value))}</strong></div>"
        for label, value in rows
    )
    st.markdown(f'<div class="facts">{items}</div>', unsafe_allow_html=True)


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
        critical_data_complete = all(v is not None and not pd.isna(v) for v in critical_scores)

        # Check the actual measured values against ECOCROP absolute bounds.
        # A fit score of 0 can also occur exactly at an inclusive boundary, so
        # using score <= 0 would incorrectly reject a crop sitting on that edge.
        def outside_range(value, lower, upper):
            if value is None or pd.isna(value) or lower is None or pd.isna(lower) or upper is None or pd.isna(upper):
                return False
            return float(value) < float(lower) or float(value) > float(upper)

        outside_absolute_limit = (
            outside_range(annual_temp, crop.get("temp_abs_min_c"), crop.get("temp_abs_max_c"))
            or outside_range(annual_rain, crop.get("annual_rain_abs_min_mm"), crop.get("annual_rain_abs_max_mm"))
            or (ph_available and outside_range(soil_ph, crop.get("ph_abs_min"), crop.get("ph_abs_max")))
            or (texture_score is not None and float(texture_score) <= 0)
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
        elif ph_available and texture_class and texture_score is None:
            eligibility = "Climate + pH screening (texture criterion unavailable)"
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
                    "Climate + pH screening (texture criterion unavailable)",
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
if "location_chosen" not in st.session_state:
    st.session_state.location_chosen = False
if "climate_notice" not in st.session_state:
    st.session_state.climate_notice = None
if "soil_notice" not in st.session_state:
    st.session_state.soil_notice = None
if "location_label" not in st.session_state:
    st.session_state.location_label = None
if "applied_place_key" not in st.session_state:
    st.session_state.applied_place_key = None
if "place_notice" not in st.session_state:
    st.session_state.place_notice = None
pending_nav = st.session_state.pop("pending_nav", None)
if pending_nav:
    st.session_state.nav_page = pending_nav
elif st.session_state.get("pending_point_load"):
    st.session_state.nav_page = "Climate Analysis"

page, reopen_location = render_topnav()

climate = data("climate_monthly")
soil = data("soil_site_summary")
profile = data("soil_profile")
suit = data("crop_suitability_assessment")
rotation = data("crop_rotation_candidates")
root = data("crop_root_zone_water")
ecological = data("crop_ecological_requirements")
catalog = data("crop_catalog")

if st.session_state.pop("pending_point_load", False):
    with st.spinner("Loading NASA POWER climate data and SoilGrids soil data for this pin..."):
        load_selected_point(ecological, catalog)
    st.rerun()


@st.dialog("Choose a location", width="large")
def location_dialog():
    """Location picker opened from the place button. A choice loads data immediately."""
    render_location_map("dialog_")


if reopen_location:
    location_dialog()

if page == "Location":
    hero(
        "Choose a location.",
        "Select a district or click the map. Climate and soil load on their own, then the Climate page opens.",
    )
    render_location_map()

    if not st.session_state.location_chosen:
        st.info(
            "No location selected yet. Choose a district from the lists, or click the map. "
            "The opening view is only a map center, not a selected field."
        )

elif page == "Climate Analysis":
    render_report_rail("right", "Soil", "Soil Analysis")
    hero(
        "Climate analysis",
        "30-year monthly climate reference derived from NASA POWER monthly observations.",
    )
    if st.session_state.location_chosen:
        render_point_facts()
        show_remote_notice(st.session_state.soil_notice)
    if not st.session_state.location_chosen:
        prompt_for_location()
        climate_to_show = pd.DataFrame()
        annual_history = pd.DataFrame()
    elif not climate_is_current():
        prompt_for_climate()
        climate_to_show = pd.DataFrame()
        annual_history = pd.DataFrame()
    else:
        climate_to_show = st.session_state.selected_climate.copy()
        st.success(
            f"Showing NASA POWER {st.session_state.climate_period} reference data for "
            f"{st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}"
        )
        annual_history = st.session_state.selected_annual_climate.copy()

    if not climate_to_show.empty:
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
    render_report_rail("left", "Climate", "Climate Analysis")
    render_report_rail("right", "Suitability", "Crop Suitability")
    hero(
        "Soil analysis",
        "Location-specific topsoil estimates retrieved from ISRIC SoilGrids, not from NASA.",
    )
    if not st.session_state.location_chosen:
        prompt_for_location()
    else:
        selected_point = current_point()
        st.write(
            f"**Selected coordinates:** {selected_point[0]:.4f}, {selected_point[1]:.4f}"
        )
        st.caption(
            "SoilGrids mean predictions for the 0–5 cm layer at the selected pin. "
            "A missing value means SoilGrids has no estimate there. It is not filled in."
        )

        show_remote_notice(st.session_state.soil_notice)

        if soil_is_current():
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
                section(
                    "Topsoil texture composition · 0–5 cm"
                    if texture_complete else "Available texture fractions · incomplete"
                )
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
            elif not (st.session_state.soil_notice and st.session_state.soil_notice.get("text")):
                st.warning(
                    "SoilGrids did not return valid topsoil texture for this pin. "
                    "This is a data gap, not a system error. Missing cells were left blank."
                )
            if soil_values.get("texture_sum_pct") is not None:
                st.caption(
                    f"Texture fractions total: {fmt(soil_values['texture_sum_pct'], 1)}%. "
                    "Small differences from 100% can occur because these are independently modelled estimates."
                )
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
                "SoilGrids has not been loaded for this pin yet. "
                "It loads together with the climate record when a place is chosen."
            )

    st.markdown(
        "[SoilGrids WCS documentation](https://docs.isric.org/globaldata/soilgrids/wcs.html)"
    )

elif page == "Crop Suitability":
    render_report_rail("left", "Soil", "Soil Analysis")
    render_report_rail("right", "Rotation", "Crop Rotation")
    hero(
        "Crop suitability screening",
        "Preliminary screening for the selected pin, using NASA POWER climate data and FAO ECOCROP requirements.",
    )
    if not st.session_state.location_chosen:
        prompt_for_location()
    elif not climate_is_current():
        prompt_for_climate()
    elif st.session_state.location_suitability.empty:
        st.info("The climate record loaded, but no crop score could be calculated for this pin.")
    else:
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
        if soil_is_current():
            current_soil = st.session_state.selected_soil or {}
            has_ph = current_soil.get("phh2o") is not None
            has_texture = bool(current_soil.get("texture_complete"))
            if has_ph and has_texture:
                coverage = "Scores include climate, SoilGrids pH, and texture where each crop has an ECOCROP range."
            elif has_ph:
                coverage = (
                    f"Scores include climate and SoilGrids pH ({fmt(current_soil.get('phh2o'))}). "
                    "Texture is incomplete for this pin and was left out."
                )
            elif has_texture:
                coverage = "Scores include climate and SoilGrids texture. pH was not available for this pin and was left out."
            else:
                coverage = "SoilGrids did not return a usable soil value for this pin. Scores use climate only."
        else:
            coverage = "Scores use the NASA POWER climate record only. SoilGrids did not add pH or texture for this pin."
        st.caption(
            coverage + " Cotton and chickpea are not scored until their ECOCROP records are added. "
            "Weights, when present: temperature 30%, rainfall 25%, pH 20%, texture 25%. "
            "Missing parts are dropped and the remaining weights are rescaled. "
            "These are screening indicators, not yield predictions."
        )
        if "Eligibility" in view.columns and view["Eligibility"].eq("Outside absolute limits").any():
            st.warning(
                "Some crops cross an ECOCROP absolute limit. Their scores stay visible, "
                "and those crops are left out of rotation candidates."
            )

elif page == "Crop Rotation":
    render_report_rail("left", "Suitability", "Crop Suitability")
    hero(
        "Crop rotation scenarios",
        "Three-year sequences built from the selected pin's crop screening.",
    )
    if not st.session_state.location_chosen:
        prompt_for_location()
    elif not climate_is_current():
        prompt_for_climate()
    elif st.session_state.location_rotations.empty:
        st.warning(
            "No three-year rotation can be built from the crops that pass the screen for this pin. "
            "SmartCrop does not fill a rotation with crops outside their absolute limits."
        )
        st.caption(
            "Only eligible annual crops are used. Olive is excluded. "
            "Each crop appears once, and the three years must belong to different crop groups."
        )
    else:
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
        st.caption(
            "These are heuristic candidates from the current screening. "
            "They are not agronomic prescriptions."
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
        "The rotation generator uses only crops passing the current screening eligibility and requires three distinct crop groups in each three-year sequence. Perennial crops such as olive are excluded. Climate-only or partial-soil rotations are explicitly preliminary when SoilGrids properties are unavailable. It does not model field history, planting calendars, pests, diseases, nutrient balances, soil carbon, irrigation, or economics.",
        "The current database has 8 catalogued crops; 6 have populated ECOCROP requirement records and are included in scoring. Cotton and chickpea are not scored until their requirement records are added.",
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
