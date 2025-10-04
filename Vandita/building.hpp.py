# house_price_dashboard.py
import os
import sys
import math
import json
from tkinter import simpledialog
import traceback
import webbrowser
from datetime import datetime
from urllib.parse import quote
import traceback
import webbrowser
from datetime import datetime
from urllib.parse import quote_plus
from urllib.request import urlopen, Request


import numpy as np
import pandas as pd

# --- Optional plotting support ---
HAS_MATPLOTLIB = True
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    HAS_MATPLOTLIB = False
    plt = None
    FigureCanvasTkAgg = None

# Try customtkinter for nicer UI; else fallback to tkinter
try:
    import customtkinter as ctk  # optional; UI will work without it
    USE_CTK = True
except Exception:
    USE_CTK = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import joblib

# ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Optional boosters: XGBoost, LightGBM, CatBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# ---------------- Synthetic dataset generator ----------------
def generate_indian_district_data(n=2500, seed=42):
    """Original synthetic generator — price depends on many features."""
    rng = np.random.RandomState(seed)
    districts = [
        "Bengaluru", "Chennai", "Hyderabad", "Delhi", "Mumbai", "Kolkata",
        "Pune", "Jaipur", "Lucknow", "Patna", "Ahmedabad", "Surat",
        "Nagpur", "Indore", "Bhopal", "Coimbatore", "Mysuru", "Mangalore",
        "Vizag", "Chandigarh", "Guwahati", "Ranchi", "Raipur", "Vadodara",
        "Thiruvananthapuram", "Amritsar", "Lucknow-East", "Agra", "Noida"
    ]
    property_types = ["Apartment", "Villa", "Independent House", "Plot"]
    furnishing = ["Furnished", "Semi-Furnished", "Unfurnished"]

    district = rng.choice(districts, size=n)
    area = rng.normal(1200, 450, size=n).clip(200, 8000)
    bedrooms = rng.randint(1, 6, size=n)
    bathrooms = rng.randint(1, 5, size=n)
    total_floors = rng.randint(1, 20, size=n)
    floor_number = np.minimum(total_floors, np.maximum(1, rng.randint(1, 21, size=n)))
    construction_year = rng.randint(1975, 2025, size=n)
    age = 2025 - construction_year
    parking_spaces = rng.randint(0, 4, size=n)
    balconies = rng.randint(0, 4, size=n)
    nearby_schools = rng.randint(0, 15, size=n)
    nearby_hospitals = rng.randint(0, 8, size=n)
    furnishing_status = rng.choice(furnishing, size=n, p=[0.2, 0.3, 0.5])
    property_type = rng.choice(property_types, size=n, p=[0.6, 0.07, 0.25, 0.08])
    has_elevator = ((total_floors >= 6) & (property_type == "Apartment")).astype(int)

    high = {"Mumbai", "Delhi", "Bengaluru"}
    mid = {"Chennai", "Hyderabad", "Pune", "Kolkata", "Vadodara", "Noida"}
    district_factor = np.array([1.6 if d in high else (1.2 if d in mid else 0.9) for d in district])

    base = area * 250
    type_factor = np.array([1.7 if t == "Villa" else (1.3 if t == "Independent House" else 1.0) for t in property_type])
    furnish_factor = np.array([1.25 if f == "Furnished" else (1.12 if f == "Semi-Furnished" else 1.0) for f in furnishing_status])
    parking_factor = 1 + (parking_spaces * 0.06)
    balcony_factor = 1 + (balconies * 0.02)

    noise = rng.normal(0, 90000, size=n)
    price = (base * district_factor * type_factor * furnish_factor * parking_factor * balcony_factor)
    price = price + (bedrooms * 45000) - (age * 750) + (nearby_schools * 1200) + (nearby_hospitals * 1800) + noise

    df = pd.DataFrame({
        "building_id": ["B" + str(i).zfill(6) for i in range(1, n + 1)],
        "city_district": district,
        "property_type": property_type,
        "area_sqft": np.round(area, 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "total_floors": total_floors,
        "floor_number": floor_number,
        "construction_year": construction_year,
        "age_years": age,
        "furnishing_status": furnishing_status,
        "parking_spaces": parking_spaces,
        "balconies": balconies,
        "nearby_schools": nearby_schools,
        "nearby_hospitals": nearby_hospitals,
        "has_elevator": has_elevator,
        "owner_type": rng.choice(["Owner", "Builder", "Rental Investment"], size=n, p=[0.7, 0.15, 0.15]),
        "listed_by": rng.choice(["Agent", "Owner", "Developer"], size=n, p=[0.5, 0.4, 0.1]),
        "price": np.round(price, 2),
    })
    return df
# ---------------- Real-data fetch helpers ----------------
def fetch_csv_or_json_url(url, timeout=20):
    """
    Try to fetch a CSV or JSON from a provided URL. If CSV-like, return DataFrame.
    If JSON, attempt to normalize into DataFrame.
    Raises ValueError on failure.
    """
    try:
        req = Request(url, headers={"User-Agent": "MergedHousePriceApp/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            # try CSV first
            try:
                s = raw.decode('utf-8', errors='replace')
                df = pd.read_csv(pd.compat.StringIO(s))
                return df
            except Exception:
                # try json
                try:
                    j = json.loads(raw.decode('utf-8', errors='replace'))
                    # If top-level has 'records' or 'data', try that
                    if isinstance(j, dict):
                        if 'records' in j and isinstance(j['records'], list):
                            df = pd.json_normalize(j['records'])
                            return df
                        # data.gov.in style
                        if 'data' in j and isinstance(j['data'], list):
                            df = pd.json_normalize(j['data'])
                            return df
                        # fallback: normalize entire dict if possible
                        df = pd.json_normalize(j)
                        return df
                    elif isinstance(j, list):
                        df = pd.json_normalize(j)
                        return df
                    else:
                        raise ValueError("Downloaded JSON had unexpected structure")
                except Exception as e_json:
                    raise ValueError(f"Failed to parse content from {url}: {e_json}")
    except Exception as e:
        raise ValueError(f"Failed to fetch URL {url}: {e}")

def fetch_from_data_gov_in(resource_id, api_key=None, limit=5000):
    """
    Helper to fetch dataset from data.gov.in API.
    data.gov.in API format (example):
      https://api.data.gov.in/resource/<resource_id>?api-key=<api_key>&format=csv&offset=0&limit=1000
    Requires you to register for an API key at data.gov.in. If api_key is None it will check
    the DATA_GOV_IN_KEY env var. If not found, raises ValueError.
    NOTE: You must provide a real resource_id for the specific dataset you want.
    """
    if api_key is None:
        api_key = os.environ.get("DATA_GOV_IN_KEY")
    if not api_key:
        raise ValueError("No DATA_GOV_IN_KEY found. Set environment variable or pass api_key.")
    if not resource_id:
        resource_id = os.environ.get("DATA_GOV_RESOURCE_ID")
    if not resource_id:
        raise ValueError("No data.gov.in resource_id provided. Set DATA_GOV_RESOURCE_ID or pass resource_id.")

    # build CSV endpoint
    base = "https://api.data.gov.in/resource/{}".format(resource_id)
    params = f"?api-key={quote_plus(api_key)}&format=csv&limit={limit}"
    url = base + params
    return fetch_csv_or_json_url(url)
# ---------------- Real-time generator (district-size based) ----------------
def generate_realtime_district_data(n=500, seed=42):
    """Generate 'real-time' style data where 'price' is determined ONLY by district size."""
    rng = np.random.RandomState(seed)
    districts = [
        "Bengaluru", "Chennai", "Hyderabad", "Delhi", "Mumbai", "Kolkata",
        "Pune", "Jaipur", "Lucknow", "Patna", "Ahmedabad", "Surat",
        "Nagpur", "Indore", "Bhopal", "Coimbatore", "Mysuru", "Mangalore",
        "Vizag", "Chandigarh", "Guwahati", "Ranchi", "Raipur", "Vadodara",
        "Thiruvananthapuram", "Amritsar", "Lucknow-East", "Agra", "Noida"
    ]
    large = {"Mumbai", "Delhi", "Bengaluru"}
    medium = {"Chennai", "Hyderabad", "Pune", "Kolkata", "Vadodara", "Noida"}
    size_map = {d: ('large' if d in large else ('medium' if d in medium else 'small')) for d in districts}
    size_base = {'large': 12000000.0, 'medium': 6500000.0, 'small': 2500000.0}
    district_choices = rng.choice(districts, size=n)
    area = rng.normal(1200, 450, size=n).clip(200, 8000)
    bedrooms = rng.randint(1, 6, size=n)
    bathrooms = rng.randint(1, 5, size=n)
    furnishing_status = rng.choice(["Furnished", "Semi-Furnished", "Unfurnished"], size=n, p=[0.2, 0.3, 0.5])
    property_type = rng.choice(["Apartment", "Villa", "Independent House", "Plot"], size=n, p=[0.6, 0.07, 0.25, 0.08])
    total_floors = rng.randint(1, 20, size=n)
    floor_number = np.minimum(total_floors, np.maximum(1, rng.randint(1, 21, size=n)))
    parking_spaces = rng.randint(0, 4, size=n)
    balconies = rng.randint(0, 4, size=n)
    construction_year = rng.randint(2000, 2025, size=n)
    age = 2025 - construction_year
    nearby_schools = rng.randint(0, 15, size=n)
    nearby_hospitals = rng.randint(0, 8, size=n)

    prices = []
    retrieved_at = []
    for d in district_choices:
        size = size_map[d]
        base = size_base[size]
        noise = rng.normal(0, base * 0.03)
        t_factor = 1.0 + (rng.normal(0, 0.002))
        price = max(10000.0, (base + noise) * t_factor)
        prices.append(round(price, 2))
        retrieved_at.append(datetime.utcnow().isoformat() + "Z")

    df = pd.DataFrame({
        "building_id": ["RT" + str(i).zfill(6) for i in range(1, n + 1)],
        "city_district": district_choices,
        "district_size": [size_map[d] for d in district_choices],
        "property_type": property_type,
        "area_sqft": np.round(area, 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "total_floors": total_floors,
        "floor_number": floor_number,
        "construction_year": construction_year,
        "age_years": age,
        "furnishing_status": furnishing_status,
        "parking_spaces": parking_spaces,
        "balconies": balconies,
        "nearby_schools": nearby_schools,
        "nearby_hospitals": nearby_hospitals,
        "has_elevator": ((total_floors >= 6) & (property_type == "Apartment")).astype(int),
        "owner_type": rng.choice(["Owner", "Builder", "Rental Investment"], size=n, p=[0.7, 0.15, 0.15]),
        "listed_by": rng.choice(["Agent", "Owner", "Developer"], size=n, p=[0.5, 0.4, 0.1]),
        "price": prices,
        "retrieved_at": retrieved_at,
    })
    return df
# ---------------- Data audit logic ----------------
def audit_dataframe(df: pd.DataFrame, verbose=True):
    res = {}
    n_rows, n_cols = df.shape
    res['basic'] = {
        'rows': int(n_rows),
        'columns': int(n_cols),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum())
    }
    num = df.select_dtypes(include=[np.number]).copy()
    num_summary = {}
    for c in num.columns:
        col = num[c].dropna()
        if col.empty:
            num_summary[c] = {'count': 0}
            continue
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        iqr = q3 - q1
        mean = float(col.mean())
        median = float(col.median())
        std = float(col.std())
        minv = float(col.min())
        maxv = float(col.max())
        num_summary[c] = dict(count=int(col.count()), mean=mean, median=median, std=std, min=minv, q1=q1, q3=q3, max=maxv)
        outliers = ((col < q1 - 3 * iqr) | (col > q3 + 3 * iqr)).sum()
        num_summary[c]['outliers_3iqr'] = int(outliers)
    res['numeric_summary'] = num_summary

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    low_variance = [c for c in df.columns if df[c].dtype != 'object' and df[c].nunique() <= max(2, int(0.01 * n_rows))]
    res['constant_columns'] = const_cols
    res['low_variance_columns'] = low_variance

    if 'building_id' in df.columns:
        ids = df['building_id'].astype(str)
        numeric_suffix = ids.str.extract(r'(\d+)$', expand=False).dropna()
        if len(numeric_suffix) > 0:
            numeric_suffix = numeric_suffix.astype(int)
            consecutive_frac = ((np.diff(np.sort(numeric_suffix)) == 1).sum()) / max(1, len(numeric_suffix)-1)
            res['id_sequential_fraction'] = float(consecutive_frac)
        else:
            res['id_sequential_fraction'] = 0.0

    if {'price', 'area_sqft'}.issubset(df.columns):
        pps = df['price'] / df['area_sqft']
        res['price_per_sqft'] = {
            'median': float(pps.median()),
            'mean': float(pps.mean()),
            'min': float(pps.min()),
            'max': float(pps.max()),
            'extreme_low_pct': float((pps < 100).mean()),
            'extreme_high_pct': float((pps > 20000).mean())
        }
        if pps.median() < 1000:
            res['pps_flag'] = 'median_pps_low'
        elif pps.median() > 50000:
            res['pps_flag'] = 'median_pps_high'
        else:
            res['pps_flag'] = 'pps_ok'

    synthetic_warnings = []
    for c in num.columns:
        col = num[c].dropna()
        if len(col) < 30:
            continue
        diffs = (col.round(0).diff().abs().dropna())
        pct_zero_diffs = float((diffs == 0).mean()) if len(diffs) > 0 else 0.0
        unique_frac = float(col.nunique() / len(col))
        if unique_frac < 0.03:
            synthetic_warnings.append(f"Column '{c}' has very low unique fraction: {unique_frac:.3f}")
        if pct_zero_diffs > 0.6:
            synthetic_warnings.append(f"Column '{c}' shows many repeated adjacent values (pct={pct_zero_diffs:.2f})")
    res['synthetic_warnings'] = synthetic_warnings

    corr_msgs = []
    if 'price' in df.columns:
        for expected in ['area_sqft', 'bedrooms']:
            if expected in df.columns and df[expected].dtype != 'object':
                try:
                    cval = df[['price', expected]].dropna().corr().iloc[0,1]
                    if abs(cval) < 0.1:
                        corr_msgs.append(f"Low correlation between price and {expected}: {cval:.3f}")
                except Exception:
                    pass
    res['correlation_messages'] = corr_msgs

    score = 0
    score -= res['basic']['missing_values'] > 0 and 1 or 0
    score -= res['basic']['duplicate_rows'] > 0 and 1 or 0
    score -= len(res['constant_columns']) > 0 and 1 or 0
    score -= len(res['synthetic_warnings']) > 0 and 1 or 0
    score -= res.get('id_sequential_fraction', 0.0) > 0.6 and 1 or 0
    try:
        pps_m = float(res['price_per_sqft']['median'])
        if 2000 <= pps_m <= 20000:
            score += 2
        elif 500 <= pps_m < 2000:
            score += 1
    except Exception:
        pass
    res['heuristic_score'] = int(score)
    res['heuristic_decision'] = 'likely_real' if score >= 1 else 'possibly_synthetic_or_noisy'
    return res

# ---------------- Helper for feature preparation ----------------
def prepare_features_for_training(df):
    X = df.drop(columns=['building_id'], errors='ignore').copy()
    if 'price' in X.columns:
        X = X.drop(columns=['price'])
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X

# ---------------- Small DataTable (Treeview) for preview ----------------
class DataTable(ttk.Frame):
    def __init__(self, master, df: pd.DataFrame, max_rows=500, **kwargs):
        super().__init__(master, **kwargs)
        self.df = df
        self.max_rows = max_rows
        self._build_grid()

    def _build_grid(self):
        cols = list(self.df.columns)
        self.tree = ttk.Treeview(self, columns=cols, show='headings', height=20)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor='center')

        vbar = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)
        hbar = ttk.Scrollbar(self, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.tree.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        for _, row in self.df.head(self.max_rows).iterrows():
            vals = [row[c] for c in cols]
            self.tree.insert('', 'end', values=vals)

    def export_excel(self, path):
        self.df.to_excel(path, index=False)

# ---------------- Graph & Stats windows (guard for matplotlib) ----------------
class GraphWindow(tk.Toplevel):
    def __init__(self, master, fig, title="Graph"):
        super().__init__(master)
        self.title(title)
        self.geometry("900x700")
        if HAS_MATPLOTLIB:
            canvas = FigureCanvasTkAgg(fig, master=self)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill='both', expand=True)
        else:
            ttk.Label(self, text="Matplotlib not available").pack(fill='both', expand=True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

class StatsWindow(tk.Toplevel):
    def __init__(self, master, df, predictions):
        super().__init__(master)
        self.title("Prediction Stats")
        self.geometry("900x700")
        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True, padx=8, pady=8)

        mean_pred = np.mean(predictions)
        median_pred = np.median(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        std_pred = np.std(predictions)
        count_pred = len(predictions)
        stats_text = (
            f"Total buildings: {count_pred}\n"
            f"Mean predicted price: {mean_pred:,.2f}\n"
            f"Median predicted price: {median_pred:,.2f}\n"
            f"Min predicted price: {min_pred:,.2f}\n"
            f"Max predicted price: {max_pred:,.2f}\n"
            f"Std deviation: {std_pred:,.2f}"
        )
        lbl = ttk.Label(frame, text=stats_text, justify='left', font=("Segoe UI", 11))
        lbl.grid(row=0, column=0, sticky='nw')

        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(predictions, bins=40)
            ax.set_title("Distribution of Predicted Prices")
            ax.set_xlabel("Predicted Price")
            ax.set_ylabel("Count")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew', pady=8)
            frame.grid_rowconfigure(1, weight=1)
            frame.grid_columnconfigure(0, weight=1)

# ---------------- Main GUI app ----------------
class MergedHousePriceApp:
    def __init__(self, root):
        self.root = root
        if USE_CTK:
            try:
                ctk.set_appearance_mode("System")
                ctk.set_default_color_theme("blue")
            except Exception:
                pass
        root.title("House Price Prediction Dashboard - Merged")
        root.geometry("1200x800")

        # state
        self.df = None
        self.model = None
        self.model_name = None
        self.train_results = None
        os.makedirs('outputs', exist_ok=True)

        # front page
        self.front_frame = ttk.Frame(self.root)
        self.front_frame.pack(fill='both', expand=True)
        self._build_front_page()

# ---------------- Main GUI app ----------------
class MergedHousePriceApp:
    def __init__(self, root):
        self.root = root
        if USE_CTK:
            try:
                ctk.set_appearance_mode("System")
                ctk.set_default_color_theme("blue")
            except Exception:
                pass
        root.title("House Price Prediction Dashboard - Merged (Real-data)")
        root.geometry("1200x860")

        # state
        self.df = None         # loaded dataset (should be real data when using CMA)
        self.model = None
        self.model_name = None
        self.train_results = None
        os.makedirs('outputs', exist_ok=True)

        # front page
        self.front_frame = ttk.Frame(self.root)
        self.front_frame.pack(fill='both', expand=True)
        self._build_front_page()

    # ---------------- Front page ----------------
    def _build_front_page(self):
        for w in self.front_frame.winfo_children():
            w.destroy()
        ttk.Label(self.front_frame, text="House Price Dashboard (Real-data mode)", font=("Helvetica", 20)).pack(pady=18)
        ttk.Label(self.front_frame, text="Data ↔︎ EDA ↔︎ Train ↔︎ Predict ↔︎ Audit ↔︎ Valuation Help", font=("Helvetica", 12)).pack(pady=6)
        btn_frame = ttk.Frame(self.front_frame)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="Enter Dashboard", command=self._enter_dashboard).grid(row=0, column=0, padx=8)
        ttk.Button(btn_frame, text="Run Audit Tests (--test)", command=self._run_cli_tests).grid(row=0, column=1, padx=8)
        ttk.Button(btn_frame, text="Quit", command=self.root.quit).grid(row=0, column=2, padx=8)
        ttk.Label(self.front_frame, text="Tip: Load a real dataset (CSV/JSON) or configure data.gov.in API to use CMA.", foreground='gray').pack(pady=10)
        status_text = f"Matplotlib: {'available' if HAS_MATPLOTLIB else 'NOT available'}  |  XGBoost: {'yes' if HAS_XGB else 'no'}  |  LightGBM: {'yes' if HAS_LGBM else 'no'}  |  CatBoost: {'yes' if HAS_CAT else 'no'}"
        ttk.Label(self.front_frame, text=status_text, font=("Courier", 10)).pack(pady=6)

    # ---------------- CLI test runner ----------------
    def _run_cli_tests(self):
        try:
            self._run_tests_internal()
            messagebox.showinfo("Tests", "All tests passed ✅")
        except AssertionError as e:
            messagebox.showerror("Tests failed", str(e))
        except Exception as e:
            messagebox.showerror("Tests error", str(e))

    def _run_tests_internal(self):
        df = pd.DataFrame({'price':[100000,200000],'area_sqft':[100,200]})
        audit = audit_dataframe(df)
        assert 'basic' in audit and 'numeric_summary' in audit and 'heuristic_decision' in audit, 'audit missing keys'
        return 0

    # ---------------- Front page ----------------
    def _build_front_page(self):
        for w in self.front_frame.winfo_children():
            w.destroy()
        ttk.Label(self.front_frame, text="House Price Dashboard", font=("Helvetica", 20)).pack(pady=18)
        ttk.Label(self.front_frame, text="Merged app: Data ↔ EDA ↔ Train ↔ Predict ↔ Audit ↔ Valuation Help", font=("Helvetica", 12)).pack(pady=6)
        btn_frame = ttk.Frame(self.front_frame)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="Enter Dashboard", command=self._enter_dashboard).grid(row=0, column=0, padx=8)
        ttk.Button(btn_frame, text="Run Audit Tests (--test)", command=self._run_cli_tests).grid(row=0, column=1, padx=8)
        ttk.Button(btn_frame, text="Quit", command=self.root.quit).grid(row=0, column=2, padx=8)
        ttk.Label(self.front_frame, text="Tip: If matplotlib is missing, plots will be disabled.", foreground='gray').pack(pady=10)
        status_text = f"Matplotlib: {'available' if HAS_MATPLOTLIB else 'NOT available'}  |  XGBoost: {'yes' if HAS_XGB else 'no'}  |  LightGBM: {'yes' if HAS_LGBM else 'no'}  |  CatBoost: {'yes' if HAS_CAT else 'no'}"
        ttk.Label(self.front_frame, text=status_text, font=("Courier", 10)).pack(pady=6)

    # ---------------- CLI test runner ----------------
    def _run_cli_tests(self):
        try:
            self._run_tests_internal()
            messagebox.showinfo("Tests", "All tests passed ✅")
        except AssertionError as e:
            messagebox.showerror("Tests failed", str(e))
        except Exception as e:
            messagebox.showerror("Tests error", str(e))

    def _run_tests_internal(self):
        df = generate_indian_district_data(n=200, seed=0)
        audit = audit_dataframe(df)
        assert 'basic' in audit and 'numeric_summary' in audit and 'heuristic_decision' in audit, 'audit missing keys'
        assert isinstance(audit['basic']['rows'], int) and audit['basic']['rows'] == 200
        df2 = df.copy(); df2['const_col'] = 1
        audit2 = audit_dataframe(df2)
        assert 'const_col' in audit2['constant_columns'] or 'const_col' in audit2.get('low_variance_columns', []), 'constant column not detected'
        df3 = pd.DataFrame({'price': [1000, 2000, 3000], 'area_sqft': [50, 60, 70]})
        audit3 = audit_dataframe(df3)
        assert 'price_per_sqft' in audit3, 'pps missing'
        return 0

    # ---------------- Dashboard & Tabs ----------------
    def _enter_dashboard(self):
        self.front_frame.pack_forget()
        self._build_notebook()

    def _build_notebook(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True)

        # create tabs and add to notebook
        self.tab_data = ttk.Frame(self.nb)
        self.tab_eda = ttk.Frame(self.nb)
        self.tab_train = ttk.Frame(self.nb)
        self.tab_predict = ttk.Frame(self.nb)
        self.tab_audit = ttk.Frame(self.nb)
        self.tab_help = ttk.Frame(self.nb)
        self.tab_alg = ttk.Frame(self.nb)

        self.nb.add(self.tab_data, text='Data')
        self.nb.add(self.tab_eda, text='EDA')
        self.nb.add(self.tab_train, text='Train')
        self.nb.add(self.tab_predict, text='Predict')
        self.nb.add(self.tab_audit, text='Audit')
        self.nb.add(self.tab_help, text='Valuation Help')
        self.nb.add(self.tab_alg, text='Algorithms & Outputs')

        # build each tab UI and logic
        self._build_data_tab()
        self._build_eda_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_audit_tab()
        self._build_help_tab()   # NEW: valuation help
        self._build_alg_tab()

    # ---------------- Data tab ----------------
    def _build_data_tab(self):
        f = self.tab_data
        top = ttk.Frame(f); top.pack(fill='x', pady=8)
        ttk.Button(top, text='Generate Sample (3k)', command=self._on_generate).pack(side='left', padx=6)
        ttk.Button(top, text='Real-Time (synthetic demo)', command=self._on_generate_realtime).pack(side='left', padx=6)
        ttk.Button(top, text='Load CSV/Excel (local real data)', command=self._on_load).pack(side='left', padx=6)
        ttk.Button(top, text='Save CSV', command=self._on_save).pack(side='left', padx=6)
        ttk.Button(top, text='Preview Top 20', command=self._show_preview).pack(side='left', padx=6)
        ttk.Button(top, text='Load Real Data (URL)', command=self._on_load_url).pack(side='left', padx=6)
        ttk.Button(top, text='Load from data.gov.in', command=self._on_load_data_gov_in).pack(side='left', padx=6)

        self.data_preview_frame = ttk.Frame(f, relief='sunken')
        self.data_preview_frame.pack(fill='both', expand=True, padx=8, pady=8)


    def _on_generate(self):
        try:
            self.df = generate_indian_district_data(n=3000)
            messagebox.showinfo('Generated', f'Generated {len(self.df)} rows (feature-based synthetic)')
            self._show_preview()
        except Exception as e:
            messagebox.showerror('Error', str(e))


    def _on_generate_realtime(self):
        try:
            self.df = generate_realtime_district_data(n=500, seed=np.random.randint(0, 10000))
            messagebox.showinfo('Real-Time generated', f'Real-time dataset generated ({len(self.df)} rows).\nPrice based ONLY on district size (large/medium/small).')
            self._show_preview()
        except Exception as e:
            messagebox.showerror('Real-time generation failed', f'{e}\n{traceback.format_exc()}')

    def _on_load(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files', '.csv'), ('Excel files', '.xlsx;*.xls')])
        if not path:
            return
        try:
            if path.lower().endswith('.csv'):
                self.df = pd.read_csv(path)
            else:
                self.df = pd.read_excel(path)
            messagebox.showinfo('Loaded', f'Loaded {os.path.basename(path)} ({len(self.df)} rows)')
            if len(self.df) < 20000:
                messagebox.showwarning('Row count', f'Loaded dataset has {len(self.df)} rows — you requested >=20000 for "real data". If you have a larger dataset, load that file instead.')
            self._show_preview()
        except Exception as e:
            messagebox.showerror('Load failed', f'{e}\n{traceback.format_exc()}')


    def _on_save(self):
        if self.df is None:
            messagebox.showwarning('No data', 'No data to save')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            messagebox.showinfo('Saved', f'Saved to {path}')
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    def _on_load(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv'), ('Excel files', '*.xlsx;*.xls')])
        if not path:
            return
        try:
            if path.lower().endswith('.csv'):
                self.df = pd.read_csv(path)
            else:
                self.df = pd.read_excel(path)
            messagebox.showinfo('Loaded', f'Loaded {os.path.basename(path)} ({len(self.df)} rows)')
            self._show_preview()
        except Exception as e:
            messagebox.showerror('Load failed', f'{e}\n{traceback.format_exc()}')

    def _on_load_url(self):
        url = simpledialog.askstring("Load from URL", "Enter a direct CSV/JSON URL (HTTPS):", parent=self.root)
        if not url:
            return
        try:
            df = fetch_csv_or_json_url(url)
            self.df = df
            messagebox.showinfo('Loaded', f'Loaded data from URL ({len(self.df)} rows)')
            self._show_preview()
        except Exception as e:
            messagebox.showerror('URL load failed', f'{e}\n{traceback.format_exc()}')

    def _on_load_data_gov_in(self):
        # ask user for resource id and optionally API key (or use env var)
        resource_id = simpledialog.askstring("data.gov.in", "Enter data.gov.in resource_id (or leave blank to use DATA_GOV_RESOURCE_ID env var):", parent=self.root)
        api_key = simpledialog.askstring("data.gov.in", "Enter data.gov.in API key (or leave blank to use DATA_GOV_IN_KEY env var):", parent=self.root)
        if not resource_id and not os.environ.get("DATA_GOV_RESOURCE_ID"):
            messagebox.showwarning("No resource_id", "You must supply a resource_id or set DATA_GOV_RESOURCE_ID environment variable.")
            return
        try:
            df = fetch_from_data_gov_in(resource_id=resource_id, api_key=api_key, limit=5000)
            self.df = df
            messagebox.showinfo('Loaded', f'Loaded data.gov.in dataset ({len(self.df)} rows)')
            self._show_preview()
        except Exception as e:
            messagebox.showerror('data.gov.in load failed', f'{e}\n{traceback.format_exc()}')

    def _on_save(self):
        if self.df is None:
            messagebox.showwarning('No data', 'No data to save')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            messagebox.showinfo('Saved', f'Saved to {path}')
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    def _show_preview(self):
        for c in self.data_preview_frame.winfo_children():
            c.destroy()
        if self.df is None:
            ttk.Label(self.data_preview_frame, text='No data loaded').pack()
            return
        table = DataTable(self.data_preview_frame, self.df, max_rows=200)
        table.pack(fill='both', expand=True)
        self.current_table = table

    # ---------------- EDA tab ----------------
    def _build_eda_tab(self):
        f = self.tab_eda
        top = ttk.Frame(f); top.pack(fill='x', pady=8)
        ttk.Button(top, text='Show Head & Describe', command=self._eda_summary).pack(side='left', padx=6)
        ttk.Button(top, text='Plot Price Histogram', command=self._eda_plot_price).pack(side='left', padx=6)
        ttk.Button(top, text='Show Correlation Heatmap', command=self._eda_plot_corr).pack(side='left', padx=6)
        self.eda_frame = ttk.Frame(f); self.eda_frame.pack(fill='both', expand=True, padx=8, pady=8)
        self.eda_text = tk.Text(self.eda_frame, height=20)
        self.eda_text.pack(fill='both', expand=True)

    def _eda_summary(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load or generate data first'); return
        s = self.df.head(10).to_string(index=False) + '\n\n' + self.df.describe(include='all').to_string()
        self.eda_text.configure(state='normal'); self.eda_text.delete('1.0', tk.END); self.eda_text.insert('1.0', s); self.eda_text.configure(state='disabled')

    def _eda_plot_price(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load or generate data first'); return
        if not HAS_MATPLOTLIB:
            messagebox.showwarning('Plotting unavailable', 'matplotlib not installed; cannot plot'); return
        fig, ax = plt.subplots(figsize=(6,3))
        try:
            if 'price' in self.df.columns:
                self.df['price'].hist(bins=40, ax=ax)
                ax.set_title('Price Distribution')
            else:
                ax.text(0.5,0.5,'No price column to plot',ha='center')
        except Exception:
            ax.text(0.5,0.5,'plot failed',ha='center')
        win = tk.Toplevel(self.root); win.title('Price Histogram')
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw(); canvas.get_tk_widget().pack(fill='both', expand=True)

    def _eda_plot_corr(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load or generate data first'); return
        if not HAS_MATPLOTLIB:
            messagebox.showwarning('Plotting unavailable', 'matplotlib not installed; cannot plot'); return
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            messagebox.showinfo('Not enough numeric', 'Need at least 2 numeric columns to compute correlation'); return
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.matshow(corr); fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90); ax.set_yticklabels(corr.columns)
        ax.set_title('Numeric Correlations (approx)')
        win = tk.Toplevel(self.root); win.title('Correlation Heatmap')
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw(); canvas.get_tk_widget().pack(fill='both', expand=True)

    # ---------------- Train tab ----------------
    def _build_train_tab(self):
        f = self.tab_train
        left = ttk.Frame(f); left.pack(side='left', fill='y', padx=8, pady=8)
        right = ttk.Frame(f); right.pack(side='right', fill='both', expand=True, padx=8, pady=8)
        ttk.Label(left, text='Choose algorithm:').pack(pady=4)
        algos = ['LinearRegression','Ridge','Lasso','RandomForest','GradientBoosting']
        if HAS_XGB: algos.append('XGBoost')
        if HAS_LGBM: algos.append('LightGBM')
        if HAS_CAT: algos.append('CatBoost')
        self.train_algo_cb = ttk.Combobox(left, values=algos)
        self.train_algo_cb.pack(pady=4)
        ttk.Label(left, text='Test size (0-1)').pack(pady=(10,2))
        self.test_size_entry = ttk.Entry(left, width=8); self.test_size_entry.insert(0,'0.2'); self.test_size_entry.pack(pady=2)
        ttk.Button(left, text='Train', command=self._train_selected).pack(pady=8)
        ttk.Button(left, text='Save Best Model', command=self._save_model).pack(pady=4)

        self.train_text = tk.Text(right, height=20)
        self.train_text.pack(fill='both', expand=True)

    def _train_selected(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load or generate data first'); return
        algo = self.train_algo_cb.get()
        if not algo:
            messagebox.showwarning('No algorithm', 'Select an algorithm'); return
        try:
            X = prepare_features_for_training(self.df)
            if 'price' not in self.df.columns:
                messagebox.showwarning('No target', 'Dataset must have a "price" column'); return
            y = self.df['price'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(self.test_size_entry.get() or 0.2), random_state=42)
            if algo == 'LinearRegression':
                model = LinearRegression()
            elif algo == 'Ridge':
                model = Ridge()
            elif algo == 'Lasso':
                model = Lasso()
            elif algo == 'RandomForest':
                model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            elif algo == 'GradientBoosting':
                model = GradientBoostingRegressor(random_state=42)
            elif algo == 'XGBoost' and HAS_XGB:
                model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
            elif algo == 'LightGBM' and HAS_LGBM:
                model = LGBMRegressor(n_estimators=200, random_state=42)
            elif algo == 'CatBoost' and HAS_CAT:
                model = CatBoostRegressor(verbose=0)
            else:
                messagebox.showerror('Unavailable', f'Algorithm {algo} not available in this environment'); return

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            self.model = model
            self.model_name = algo
            self.train_results = {'rmse': rmse, 'r2': r2, 'mae': mae, 'y_test': y_test, 'y_pred': preds, 'features': X.columns.tolist()}
            self.train_text.delete('1.0', tk.END)
            self.train_text.insert(tk.END, f'Trained {algo}\nRMSE: {rmse:.2f}\nR2: {r2:.3f}\nMAE: {mae:.2f}\n')
            outdir = 'outputs'; os.makedirs(outdir, exist_ok=True)
            if HAS_MATPLOTLIB:
                try:
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.scatter(y_test, preds, alpha=0.4)
                    mn, mx = min(min(y_test), min(preds)), max(max(y_test), max(preds))
                    ax.plot([mn, mx], [mn, mx], 'r--')
                    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title(f'Actual vs Predicted - {algo}')
                    ppath = os.path.join(outdir, f'actual_vs_pred_{algo}.png')
                    fig.tight_layout(); fig.savefig(ppath); plt.close(fig)
                    self.train_text.insert(tk.END, f'Saved plot: {ppath}\n')
                except Exception:
                    pass
            try:
                if hasattr(model, 'feature_importances_'):
                    fi = model.feature_importances_
                    feat_names = X.columns.tolist()
                    idx = np.argsort(fi)[-20:]
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.barh(np.array(feat_names)[idx], fi[idx])
                    ax.set_title(f'Feature Importance - {algo}')
                    fipath = os.path.join(outdir, f'feature_importance_{algo}.png')
                    fig.tight_layout(); fig.savefig(fipath); plt.close(fig)
                    self.train_text.insert(tk.END, f'Saved feature importance: {fipath}\n')
            except Exception:
                pass
            try:
                joblib.dump({'model': model, 'features': X.columns.tolist()}, os.path.join(outdir, f'model_{algo}.joblib'))
                self.train_text.insert(tk.END, f'Model saved to outputs/model_{algo}.joblib\n')
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror('Training failed', f'{e}\n{traceback.format_exc()}')

    def _save_model(self):
        if self.model is None:
            messagebox.showwarning('No model', 'Train a model first'); return
        path = filedialog.asksaveasfilename(defaultextension='.joblib', filetypes=[('Joblib','*.joblib')])
        if not path:
            return
        try:
            joblib.dump({'model': self.model, 'features': self.train_results['features']}, path)
            messagebox.showinfo('Saved', f'Model saved to {path}')
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    # ---------------- Predict tab ----------------
    def _build_predict_tab(self):
        f = self.tab_predict
        left = ttk.Frame(f); left.pack(side='left', fill='y', padx=8, pady=8)
        right = ttk.Frame(f); right.pack(side='right', fill='both', expand=True, padx=8, pady=8)
        ttk.Label(left, text='Enter feature values:').pack(pady=4)
        self.predict_entries = {}
        for fname in ['area_sqft','bedrooms','bathrooms','age_years','parking_spaces','balconies','nearby_schools','nearby_hospitals']:
            ttk.Label(left, text=fname).pack(); ent = ttk.Entry(left); ent.pack(); self.predict_entries[fname] = ent
        self.predict_entries['city_district'] = ttk.Combobox(left, values=["Bengaluru","Chennai","Hyderabad","Delhi","Mumbai","Kolkata","Pune","Noida"])
        ttk.Label(left, text='city_district').pack(); self.predict_entries['city_district'].pack()
        self.predict_entries['property_type'] = ttk.Combobox(left, values=["Apartment","Villa","Independent House","Plot"]); ttk.Label(left,text='property_type').pack(); self.predict_entries['property_type'].pack()
        self.predict_entries['furnishing_status'] = ttk.Combobox(left, values=["Furnished","Semi-Furnished","Unfurnished"]); ttk.Label(left,text='furnishing_status').pack(); self.predict_entries['furnishing_status'].pack()

        ttk.Button(left, text='Predict (best trained model)', command=self._predict_instance).pack(pady=8)
        ttk.Button(left, text='Auto-fill districts from data', command=self._populate_districts_from_data).pack(pady=4)

        self.pred_text = tk.Text(right, height=20)
        self.pred_text.pack(fill='both', expand=True)

    def _populate_districts_from_data(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load data first'); return
        if 'city_district' in self.df.columns:
            vals = sorted(self.df['city_district'].dropna().unique().tolist())
            self.predict_entries['city_district']['values'] = vals
            messagebox.showinfo('Districts loaded', f'Loaded {len(vals)} unique districts into dropdown')
        else:
            messagebox.showwarning('No column', 'Data has no city_district column')

    def _predict_instance(self):
        if self.model is None:
            messagebox.showwarning('No model', 'Train a model first'); return
        if self.df is None:
            messagebox.showwarning('No data', 'Load/generate data used for training first'); return
        try:
            rec = {}
            for k, widget in self.predict_entries.items():
                val = widget.get().strip()
                if val == '':
                    rec[k] = np.nan
                else:
                    if k in ['area_sqft','bedrooms','bathrooms','age_years','parking_spaces','balconies','nearby_schools','nearby_hospitals']:
                        rec[k] = float(val)
                    else:
                        rec[k] = val
            df_new = pd.DataFrame([rec])
            X_all = prepare_features_for_training(self.df)
            df_new_d = pd.get_dummies(df_new)
            df_new_d = df_new_d.reindex(columns=X_all.columns, fill_value=0)
            pred = self.model.predict(df_new_d)[0]
            self.pred_text.delete('1.0', tk.END)
            self.pred_text.insert(tk.END, f'Predicted price: {pred:,.2f}\nModel: {self.model_name}')
        except Exception as e:
            messagebox.showerror('Prediction error', f'{e}\n{traceback.format_exc()}')

    # ---------------- Audit tab ----------------
    def _build_audit_tab(self):
        f = self.tab_audit
        left = ttk.Frame(f); left.pack(side='left', fill='y', padx=8, pady=8)
        right = ttk.Frame(f); right.pack(side='right', fill='both', expand=True, padx=8, pady=8)
        ttk.Button(left, text='Run Audit', command=self._run_audit).pack(pady=6)
        ttk.Button(left, text='Show Audit Plots', command=self._show_audit_plots).pack(pady=6)
        ttk.Button(left, text='Explain Outputs', command=self._explain_outputs).pack(pady=6)
        self.audit_text = tk.Text(right)
        self.audit_text.pack(fill='both', expand=True)

    def _run_audit(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load or generate data first'); return
        try:
            audit = audit_dataframe(self.df)
            self.audit_text.configure(state='normal'); self.audit_text.delete('1.0', tk.END); self.audit_text.insert('1.0', json.dumps(audit, indent=2)); self.audit_text.configure(state='disabled')
        except Exception as e:
            messagebox.showerror('Audit failed', f'{e}\n{traceback.format_exc()}')

    def _show_audit_plots(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load or generate data first'); return
        if not HAS_MATPLOTLIB:
            messagebox.showwarning('Matplotlib missing', 'Install matplotlib to view plots'); return
        top = tk.Toplevel(self.root); top.title('Audit Plots'); top.geometry('900x700')
        try:
            fig1, ax1 = plt.subplots(figsize=(6,3))
            if 'price' in self.df.columns:
                self.df['price'].hist(bins=40, ax=ax1); ax1.set_title('Price distribution')
            else:
                ax1.text(0.5,0.5,'No price column',ha='center')
            c1 = FigureCanvasTkAgg(fig1, master=top); c1.draw(); c1.get_tk_widget().pack(fill='both', expand=False)
        except Exception:
            pass
        try:
            if {'price','area_sqft'}.issubset(self.df.columns):
                fig2, ax2 = plt.subplots(figsize=(6,3))
                pps = (self.df['price']/self.df['area_sqft']).replace([np.inf,-np.inf], np.nan).dropna()
                ax2.hist(pps, bins=40); ax2.set_title('Price per sqft distribution')
                c2 = FigureCanvasTkAgg(fig2, master=top); c2.draw(); c2.get_tk_widget().pack(fill='both', expand=False)
        except Exception:
            pass

    def _explain_outputs(self):
        expl = (
            "Training outputs:\n"
            "- actual_vs_pred_<Model>.png -> actual vs predicted scatter (ideal: near diagonal)\n"
            "- feature_importance_<Model>.png -> for tree models\n"
            "- model_<Model>.joblib -> saved trained model\n"
            "Metrics: RMSE (lower better), MAE (lower better), R2 (closer to 1 better)\n"
            "\nNote: 'Real-Time' generated datasets have 'price' based ONLY on district_size column."
        )
        messagebox.showinfo('Explain outputs', expl)

    # ---------------- Valuation Help tab (NEW) ----------------
    def _build_help_tab(self):
        f = self.tab_help
        frame = ttk.Frame(f)
        frame.pack(fill='both', expand=True, padx=12, pady=12)

        # Instruction text (your steps)
        help_text = (
            "To get an accurate and up-to-date valuation, take one or more of the following steps:\n\n"
            "1) Contact a local real estate agent: A local broker can perform a Comparative Market Analysis (CMA) based on recent sales of similar properties in the same area to give you a detailed and accurate price estimate.\n\n"
            "2) Use online real estate portals: Websites like MagicBricks and 99acres have databases of property listings and can provide a general price estimate based on location, size, and other details.\n\n"
            "3) Obtain the property address: With the property's address, you can search online listings for the specific apartment building or neighborhood to find more current pricing data.\n\n"
            "Use the controls below to open portals or search for an address. You can also save contact notes (with phone/email) and run a quick CMA from your loaded dataset."
        )
        txt = tk.Text(frame, height=10, wrap='word')
        txt.insert('1.0', help_text)
        txt.configure(state='disabled')
        txt.grid(row=0, column=0, columnspan=4, sticky='nsew', pady=(0,8))

        # Buttons for portals (direct portal pages)
        ttk.Button(frame, text='Open MagicBricks', command=lambda: webbrowser.open("https://www.magicbricks.com")).grid(row=1, column=0, padx=6, pady=4, sticky='ew')
        ttk.Button(frame, text='Open 99acres', command=lambda: webbrowser.open("https://www.99acres.com")).grid(row=1, column=1, padx=6, pady=4, sticky='ew')
        ttk.Button(frame, text='Open NoBroker', command=lambda: webbrowser.open("https://www.nobroker.in")).grid(row=1, column=2, padx=6, pady=4, sticky='ew')
        ttk.Button(frame, text='Open Housing', command=lambda: webbrowser.open("https://housing.com")).grid(row=1, column=3, padx=6, pady=4, sticky='ew')

        # Address search
        ttk.Label(frame, text='Property address / search terms:').grid(row=2, column=0, columnspan=4, sticky='w', pady=(10,2))
        self.address_entry = ttk.Entry(frame)
        self.address_entry.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(0,6))
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=3, sticky='e', padx=(6,0))
        ttk.Button(btn_frame, text='Search MagicBricks', command=self._search_address_magicbricks).pack(fill='x')
        ttk.Button(frame, text='Search 99acres', command=self._search_address_99acres).grid(row=4, column=0, pady=(0,6), sticky='ew')
        ttk.Button(frame, text='Search NoBroker', command=self._search_address_nobroker).grid(row=4, column=1, pady=(0,6), sticky='ew')
        ttk.Button(frame, text='Search Housing', command=self._search_address_housing).grid(row=4, column=2, pady=(0,6), sticky='ew')

        # Contact capture
        ttk.Label(frame, text='Your name:').grid(row=5, column=0, sticky='w', pady=(8,2))
        self.contact_name = ttk.Entry(frame)
        self.contact_name.grid(row=5, column=1, sticky='ew', padx=(4,8))
        ttk.Label(frame, text='Phone:').grid(row=5, column=2, sticky='w', pady=(8,2))
        self.contact_phone = ttk.Entry(frame)
        self.contact_phone.grid(row=5, column=3, sticky='ew')
        ttk.Label(frame, text='Email:').grid(row=6, column=0, sticky='w', pady=(4,2))
        self.contact_email = ttk.Entry(frame)
        self.contact_email.grid(row=6, column=1, sticky='ew', padx=(4,8))
        ttk.Label(frame, text='Note:').grid(row=6, column=2, sticky='w', pady=(4,2))
        self.contact_note = ttk.Entry(frame)
        self.contact_note.grid(row=6, column=3, sticky='ew')

        ttk.Button(frame, text='Save Contact Agent Note', command=self._save_contact_note).grid(row=7, column=0, columnspan=4, pady=(8,4), sticky='ew')

        # CMA helper UI
        ttk.Separator(frame, orient='horizontal').grid(row=8, column=0, columnspan=4, sticky='ew', pady=(8,8))
        ttk.Label(frame, text='Quick CMA (median price-per-sqft by district)').grid(row=9, column=0, columnspan=4, sticky='w')
        ttk.Label(frame, text='District:').grid(row=10, column=0, sticky='w', pady=(6,2))
        self.cma_district = ttk.Combobox(frame, values=[])
        self.cma_district.grid(row=10, column=1, sticky='ew', padx=(4,8))
        ttk.Label(frame, text='Area (sqft):').grid(row=10, column=2, sticky='w', pady=(6,2))
        self.cma_area = ttk.Entry(frame)
        self.cma_area.grid(row=10, column=3, sticky='ew')
        ttk.Button(frame, text='Compute CMA estimate', command=self._compute_cma_estimate).grid(row=11, column=0, columnspan=4, pady=(6,8), sticky='ew')

        self.cma_text = tk.Text(frame, height=6, wrap='word')
        self.cma_text.grid(row=12, column=0, columnspan=4, sticky='nsew', pady=(0,8))

        frame.grid_rowconfigure(12, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_columnconfigure(3, weight=1)

    def _search_address_magicbricks(self):
        q = self.address_entry.get().strip()
        if not q:
            messagebox.showwarning('No address', 'Enter an address or search terms first')
            return
        # best-effort direct portal search URLs (may change with portal)
        url = "https://www.magicbricks.com/search/property?keyword=" + quote(q)
        webbrowser.open(url)

    def _search_address_99acres(self):
        q = self.address_entry.get().strip()
        if not q:
            messagebox.showwarning('No address', 'Enter an address or search terms first')
            return
        url = "https://www.99acres.com/search?searchtype=QS&keyword=" + quote(q)
        webbrowser.open(url)

    def _search_address_nobroker(self):
        q = self.address_entry.get().strip()
        if not q:
            messagebox.showwarning('No address', 'Enter an address or search terms first')
            return
        url = "https://www.nobroker.in/property/sale?search_location=" + quote(q)
        webbrowser.open(url)

    def _search_address_housing(self):
        q = self.address_entry.get().strip()
        if not q:
            messagebox.showwarning('No address', 'Enter an address or search terms first')
            return
        url = "https://housing.com/in/buy/search?query=" + quote(q)
        webbrowser.open(url)

    def _save_contact_note(self):
        name = self.contact_name.get().strip()
        phone = self.contact_phone.get().strip()
        email = self.contact_email.get().strip()
        note = self.contact_note.get().strip()
        if not note:
            note = "Contacted agent (no note entered)"
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        os.makedirs('outputs', exist_ok=True)
        path = os.path.join('outputs', f'contact_note_{ts}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp (UTC): {datetime.utcnow().isoformat()}Z\n")
            f.write(f"Name: {name}\n")
            f.write(f"Phone: {phone}\n")
            f.write(f"Email: {email}\n")
            f.write(f"Note: {note}\n")
            f.write("Action: Contact local real estate agent or search portals for valuations.\n")
        messagebox.showinfo('Saved', f'Contact note saved to {path}')

    # ---------------- CMA helper ----------------
    def _compute_cma_estimate(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Load a real dataset first (CSV/Excel or Fetch URL).'); return
        if not {'price','area_sqft','city_district'}.issubset(self.df.columns):
            messagebox.showwarning('Missing columns', 'Dataset must contain columns: price, area_sqft, city_district'); return
        try:
            df = self.df.copy()
            # filter out invalid rows
            df = df[(df['area_sqft'] > 0) & (df['price'] > 0)]
            df['pps'] = df['price'] / df['area_sqft']
            grp = df.groupby('city_district')['pps'].median().reset_index().rename(columns={'pps':'median_pps'})
            # populate combobox choices with districts
            districts = sorted(grp['city_district'].tolist())
            self.cma_district['values'] = districts

            selected = self.cma_district.get().strip()
            area_val = self.cma_area.get().strip()
            if selected == '' or area_val == '':
                # show median table and instructions
                s = "Median price-per-sqft by district (from loaded data):\n\n"
                s += grp.sort_values('median_pps', ascending=False).to_string(index=False, float_format='{:,.0f}'.format)
                self.cma_text.delete('1.0', tk.END); self.cma_text.insert('1.0', s); return

            area = float(area_val)
            row = grp[grp['city_district'] == selected]
            if row.empty:
                messagebox.showwarning('District not found', f'District "{selected}" not found in data. Showing full median table instead.')
                s = grp.sort_values('median_pps', ascending=False).to_string(index=False, float_format='{:,.0f}'.format)
                self.cma_text.delete('1.0', tk.END); self.cma_text.insert('1.0', s); return
            median_pps = float(row['median_pps'].iloc[0])
            estimate = median_pps * area
            s = (
                f"CMA Quick Estimate\nDistrict: {selected}\n"
                f"Median price-per-sqft (from dataset): {median_pps:,.2f}\n"
                f"Area (sqft): {area:,.1f}\n"
                f"Estimated price: {estimate:,.2f}\n\n"
                "Note: This is a quick median-based estimate. For an accurate valuation, perform a local CMA with comparable sales."
            )
            self.cma_text.delete('1.0', tk.END); self.cma_text.insert('1.0', s)
        except Exception as e:
            messagebox.showerror('CMA failed', f'{e}\n{traceback.format_exc()}')

    # ---------------- Algorithms tab ----------------
    def _build_alg_tab(self):
        f = self.tab_alg
        text = "Available algorithms in this environment:\n- LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting"
        if HAS_XGB: text += ", XGBoost"
        if HAS_LGBM: text += ", LightGBM"
        if HAS_CAT: text += ", CatBoost"
        text += "\n\nArtifacts are saved to the outputs/ folder."
        ttk.Label(f, text=text, justify='left').pack(fill='both', padx=8, pady=8)

# ------------- main -------------
def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        try:
            df = generate_indian_district_data(n=200, seed=0)
            audit = audit_dataframe(df)
            assert 'basic' in audit and 'numeric_summary' in audit and 'heuristic_decision' in audit
            print('Tests OK')
            return
        except AssertionError as e:
            print('Tests failed:', e); return

    if USE_CTK:
        try:
            root = ctk.CTk()
        except Exception:
            root = tk.Tk()
    else:
        root = tk.Tk()
    app = MergedHousePriceApp(root) # main app
    root.mainloop()

if __name__ == '__main__':
    main()
