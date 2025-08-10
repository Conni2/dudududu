"""
F1 Podium Predictor — Streamlit App (Dual Global clean+mixed + LightGBM + MC)
- 메타테이블 없이도 실행 가능
- 좌측 사이드바에서 GP/시즌/시뮬 파라미터를 조정하고 Predict 버튼으로 결과 표시
- 내부 로직은 기존 Step 2(Quantile + OOF 기반 불확실성) 유지

실행:
    streamlit run app.py
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import datetime as dt
import requests
import numpy as np
import pandas as pd
import fastf1
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import streamlit as st

# ========================= Page Setup =========================
st.set_page_config(page_title="F1 Podium Predictor", layout="wide")
st.title("🏁 F1 Podium Predictor")
st.caption("Dual Global (clean + mixed) · LightGBM + Quantile · Monte Carlo simulation · Weather-ready")

# ========================= Config =========================
@dataclass
class CVConfig:
    n_splits: int = 5

@dataclass
class ModelConfig:
    objective: str = "regression"
    learning_rate: float = 0.05
    n_estimators: int = 600
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 50
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8

@dataclass
class QuantileConfig:
    use_quantile: bool = True
    quantiles: Tuple[float, ...] = (0.2, 0.5, 0.8)

@dataclass
class SimulationConfig:
    n_runs: int = 1500
    rng_seed: int = 42

@dataclass
class AppConfig:
    cv: CVConfig = CVConfig()
    model: ModelConfig = ModelConfig()
    quant: QuantileConfig = QuantileConfig()
    sim: SimulationConfig = SimulationConfig()

APP_CONFIG = AppConfig()

# ========================= Helpers =========================
# --- optional: simple track coordinate hints (expand later)
TRACK_COORDS = {
    "Monaco": (43.7347, 7.4206),
    "Saudi Arabia": (21.6319, 39.1044),
    "Bahrain": (26.0325, 50.5106),
    "Japan": (34.8431, 136.5419),
    "China": (31.3389, 121.2219),
    "Miami": (25.9580, -80.2389),
    "Imola": (44.3439, 11.7167),
    "Monza": (45.6214, 9.2811),
    "Silverstone": (52.0733, -1.0140),
}

fastf1.Cache.enable_cache("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

@st.cache_data(show_spinner=False)
def _find_round_in_schedule(season: int, track_key: str) -> int:
    try:
        sched = fastf1.get_event_schedule(season)
        m = sched[(sched.get("Location", "").astype(str).str.contains(track_key, case=False, na=False)) |
                  (sched.get("EventName", "").astype(str).str.contains(track_key, case=False, na=False))]
        if len(m):
            return int(m.iloc[0]["RoundNumber"])
    except Exception:
        pass
    return -1

# ========================= Data Loading & Cleaning =========================
@st.cache_data(show_spinner=False)
def _coalesce_coords(track_key: str, lat_in: Optional[float], lon_in: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if lat_in is not None and lon_in is not None:
        return lat_in, lon_in
    for k, (la, lo) in TRACK_COORDS.items():
        if k.lower() in track_key.lower():
            return la, lo
    return None, None

@st.cache_data(show_spinner=False)
def _parse_date(date_str: str) -> dt.date:
    try:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return dt.date.today()

@st.cache_data(show_spinner=False)
def fetch_weather_open_meteo(lat: float, lon: float, date_str: str, mode: str = "forecast") -> Dict[str, float]:
    """Return {"temp_c": float, "rain_prob": float in 0..1}. Uses Open-Meteo (no API key)."""
    d = _parse_date(date_str)
    if mode == "forecast":
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,precipitation_probability&timezone=auto"
            f"&start_date={d.isoformat()}&end_date={d.isoformat()}"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json()
        hours = js.get("hourly", {}).get("time", [])
        temps = js.get("hourly", {}).get("temperature_2m", [])
        pops = js.get("hourly", {}).get("precipitation_probability", [])
        # choose 14:00 local if available, else use daily mean
        target = f"{d.isoformat()}T14:00"
        if target in hours:
            i = hours.index(target)
            temp_c = float(temps[i]) if i < len(temps) else float(np.nan)
            pop = float(pops[i]) / 100.0 if i < len(pops) else float(np.nan)
        else:
            temp_c = float(np.nanmean(temps)) if temps else float("nan")
            pop = float(np.nanmean(pops))/100.0 if pops else float("nan")
        return {"temp_c": temp_c, "rain_prob": max(0.0, min(1.0, pop))}
    else:
        # past 3 days average (d-3 .. d-1)
        start = (d - dt.timedelta(days=3)).isoformat()
        end = (d - dt.timedelta(days=1)).isoformat()
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,precipitation_probability&timezone=auto"
            f"&start_date={start}&end_date={end}"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json()
        temps = js.get("hourly", {}).get("temperature_2m", [])
        pops = js.get("hourly", {}).get("precipitation_probability", [])
        temp_c = float(np.nanmean(temps)) if temps else float("nan")
        pop = float(np.nanmean(pops))/100.0 if pops else float("nan")
        return {"temp_c": temp_c, "rain_prob": max(0.0, min(1.0, pop))}

@st.cache_data(show_spinner=True)
def load_sessions(track_key: str, seasons: Tuple[int, ...]) -> pd.DataFrame:
    recs = []
    for season in seasons:
        rnd = _find_round_in_schedule(season, track_key)
        for sess in ("Q", "R"):
            try:
                session = fastf1.get_session(season, rnd if rnd != -1 else track_key, sess)
                session.load()
                laps = session.laps.copy()
                laps["season"], laps["session"] = season, sess
                laps["circuit_id"] = getattr(session.event, "EventName", track_key)
                laps["round"] = getattr(session.event, "RoundNumber", np.nan)
                recs.append(laps)
            except Exception:
                continue
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def clean_laps(df_laps: pd.DataFrame, mode: str = "clean") -> pd.DataFrame:
    df = df_laps.copy()
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns:
            df = df[~df[col].isna()]
    for pit_col in ("PitInTime", "PitOutTime"):
        if pit_col in df.columns:
            df = df[df[pit_col].isna()]
    if "TrackStatus" in df.columns:
        if mode == "clean":
            df = df[(df["TrackStatus"].astype(str) == "1") | (df["TrackStatus"].isna())]
        else:
            df = df[df["TrackStatus"].astype(str).isin(["1", "4"]) | (df["TrackStatus"].isna())]
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns and np.issubdtype(df[col].dtype, np.timedelta64):
            df[f"{col}_s"] = df[col].dt.total_seconds()
    if "LapTime_s" in df.columns:
        def _trim(g: pd.DataFrame) -> pd.DataFrame:
            q1, q3 = g["LapTime_s"].quantile([0.25, 0.75])
            iqr = q3 - q1
            k = 2.0 if mode == "mixed" else 1.5
            lo, hi = q1 - k * iqr, q3 + k * iqr
            return g[(g["LapTime_s"] >= lo) & (g["LapTime_s"] <= hi)]
        df = df.groupby("Driver", group_keys=False).apply(_trim)
    df["data_mode"] = mode
    return df

# ========================= Feature Engineering =========================
def build_driver_race_table(df_laps: pd.DataFrame) -> pd.DataFrame:
    df = df_laps.copy()
    grp = ["season", "round", "circuit_id", "Driver"]
    agg = df.groupby(grp).agg(
        pace_median_s=("LapTime_s", "median"),
        s1_mean_s=("Sector1Time_s", "mean"),
        s2_mean_s=("Sector2Time_s", "mean"),
        s3_mean_s=("Sector3Time_s", "mean"),
        laps_count=("LapNumber", "count"),
    ).reset_index()
    agg["sector_sum_s"] = agg[["s1_mean_s", "s2_mean_s", "s3_mean_s"]].sum(axis=1)
    agg["race_id"] = agg["season"].astype(str) + "-" + agg["round"].astype(str) + "-" + agg["circuit_id"].astype(str)
    return agg


def add_quali_features(df_driver_race: pd.DataFrame, df_laps_all: pd.DataFrame) -> pd.DataFrame:
    df_q = df_laps_all[df_laps_all["session"] == "Q"].copy()
    if len(df_q) == 0:
        return df_driver_race
    df_q["LapTime_s"] = df_q["LapTime"].dt.total_seconds()
    best = df_q.groupby(["season","round","circuit_id","Driver"])['LapTime_s'].min().reset_index(name='best_q_s')
    pole = df_q.groupby(["season","round","circuit_id"])['LapTime_s'].min().reset_index().rename(columns={'LapTime_s':'pole_s'})
    feat = best.merge(pole, on=["season","round","circuit_id"], how="left")
    feat["gap_to_pole_s"] = feat["best_q_s"] - feat["pole_s"]
    return df_driver_race.merge(feat, on=["season","round","circuit_id","Driver"], how="left")


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "gap_to_pole_s", "best_q_s",
        "sector_sum_s", "s1_mean_s", "s2_mean_s", "s3_mean_s",
        # weather (optional)
        "temp_c", "rain_prob",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep + ["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"]]

# ========================= Models (LightGBM + Quantile) =========================
class GlobalRegressor:
    """LightGBM global regressor with optional quantile heads and OOF residual stats."""
    def __init__(self, cfg: ModelConfig, qcfg: QuantileConfig):
        self.cfg = cfg
        self.qcfg = qcfg
        self.model = lgb.LGBMRegressor(
            objective=cfg.objective,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            num_leaves=self.cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_data_in_leaf,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=42,
        )
        self.imputer = SimpleImputer(strategy="median")
        self.q_models: Dict[float, lgb.LGBMRegressor] = {}
        self.resid_std_global_: float = 0.25
        self.resid_std_by_driver_: Dict[str, float] = {}

    def _fit_quantiles(self, Xtr, ytr):
        self.q_models = {}
        if not self.qcfg.use_quantile:
            return
        for q in self.qcfg.quantiles:
            qm = lgb.LGBMRegressor(
                objective='quantile', alpha=q,
                learning_rate=self.cfg.learning_rate,
                n_estimators=self.cfg.n_estimators,
                num_leaves=self.cfg.num_leaves,
                max_depth=self.cfg.max_depth,
                min_child_samples=self.cfg.min_data_in_leaf,
                reg_alpha=self.cfg.reg_alpha,
                reg_lambda=self.cfg.reg_lambda,
                subsample=self.cfg.subsample,
                colsample_bytree=self.cfg.colsample_bytree,
                random_state=42,
            )
            qm.fit(Xtr, ytr)
            self.q_models[q] = qm

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
            drivers: Optional[pd.Series] = None) -> Dict[str, Any]:
        X_imp = self.imputer.fit_transform(X)
        gkf = GroupKFold(n_splits=APP_CONFIG.cv.n_splits)
        oof = np.zeros_like(y, dtype=float)
        maes = []
        for tr, te in gkf.split(X_imp, y, groups):
            self.model.fit(X_imp[tr], y.iloc[tr])
            pred = self.model.predict(X_imp[te])
            oof[te] = pred
            maes.append(mean_absolute_error(y.iloc[te], pred))
        self.cv_score_ = float(np.mean(maes))
        resid = y.values - oof
        self.resid_std_global_ = float(np.std(resid)) if len(resid) else 0.25
        if drivers is not None:
            df_res = pd.DataFrame({"Driver": drivers.values, "resid": resid})
            self.resid_std_by_driver_ = df_res.groupby("Driver")["resid"].std().dropna().to_dict()
        self.model.fit(X_imp, y)
        self._fit_quantiles(X_imp, y)
        return {"cv_mae": self.cv_score_, "resid_std": self.resid_std_global_}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.imputer.transform(X))

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        if not self.q_models:
            return {}
        X_imp = self.imputer.transform(X)
        return {q: m.predict(X_imp) for q, m in self.q_models.items()}

    def sigma_from_quantiles(self, q20: np.ndarray, q80: np.ndarray) -> np.ndarray:
        denom = 2 * 0.841621233
        spread = np.maximum(q80 - q20, 1e-6)
        return spread / denom

    def sigma_for_drivers(self, drivers: pd.Series, default: Optional[float] = None) -> np.ndarray:
        if default is None:
            default = self.resid_std_global_
        return np.array([self.resid_std_by_driver_.get(d, default) for d in drivers.values], dtype=float)

class DualGlobal:
    def __init__(self, cfg: ModelConfig, qcfg: QuantileConfig, w_mixed: float = 0.6):
        self.A = GlobalRegressor(cfg, qcfg)
        self.B = GlobalRegressor(cfg, qcfg)
        self.w_mixed = float(np.clip(w_mixed, 0.0, 1.0))

    def fit(self,
            X_clean: pd.DataFrame, y_clean: pd.Series, g_clean: np.ndarray, d_clean: pd.Series,
            X_mixed: pd.DataFrame, y_mixed: pd.Series, g_mixed: np.ndarray, d_mixed: pd.Series) -> Dict[str, Any]:
        sA = self.A.fit(X_clean, y_clean, g_clean, drivers=d_clean)
        sB = self.B.fit(X_mixed, y_mixed, g_mixed, drivers=d_mixed)
        return {"clean_cv_mae": sA["cv_mae"], "mixed_cv_mae": sB["cv_mae"], "w_mixed": self.w_mixed}

    def predict_blended(self, X_clean: pd.DataFrame, X_mixed: pd.DataFrame) -> np.ndarray:
        ya = self.A.predict(X_clean)
        yb = self.B.predict(X_mixed)
        return self.w_mixed * yb + (1 - self.w_mixed) * ya

    def predict_quantiles_blended(self, X_clean: pd.DataFrame, X_mixed: pd.DataFrame) -> Dict[float, np.ndarray]:
        qa = self.A.predict_quantiles(X_clean)
        qb = self.B.predict_quantiles(X_mixed)
        keys = set(qa.keys()) & set(qb.keys())
        out = {}
        for k in keys:
            out[k] = self.w_mixed * qb[k] + (1 - self.w_mixed) * qa[k]
        return out

    def sigma_combined(self, drivers: pd.Series, q_blend: Dict[float, np.ndarray]) -> np.ndarray:
        sig_q = None
        if 0.2 in q_blend and 0.8 in q_blend:
            sig_q = self.A.sigma_from_quantiles(q_blend[0.2], q_blend[0.8])
        sig_res_A = self.A.sigma_for_drivers(drivers)
        sig_res_B = self.B.sigma_for_drivers(drivers)
        sig_res = np.maximum(sig_res_A, sig_res_B)
        if sig_q is None:
            return sig_res
        return np.maximum(sig_q, sig_res)

# ========================= MC Simulator =========================
class PodiumResult(Tuple[pd.DataFrame, pd.DataFrame]):
    pass

def monte_carlo_podium(drivers: pd.Series,
                       base_time_s: np.ndarray,
                       pace_std_s: np.ndarray,
                       laps: int,
                       pit_loss_s: float,
                       pit_count_mean: float,
                       sc_rate: float,
                       team_order_prob: float,
                       n_runs: int = APP_CONFIG.sim.n_runs,
                       seed: int = APP_CONFIG.sim.rng_seed) -> Any:
    rng = np.random.default_rng(seed)
    n = len(drivers)
    total_times = np.zeros((n_runs, n), dtype=float)
    for r in range(n_runs):
        pace_noise = rng.normal(0.0, pace_std_s, size=(n,))
        pit_counts = rng.poisson(lam=max(pit_count_mean, 0.0), size=(n,))
        sc_flag = rng.random() < sc_rate
        sc_penalty = rng.normal(0.0, 3.0) if sc_flag else 0.0
        times = (base_time_s + pace_noise) * laps + pit_counts * pit_loss_s + sc_penalty
        if team_order_prob > 0 and rng.random() < team_order_prob:
            idx = np.argsort(times)
            if len(idx) >= 2:
                i0, i1 = idx[0], idx[1]
                times[i0], times[i1] = times[i1], times[i0]
        total_times[r, :] = times
    mean_time = total_times.mean(axis=0)
    ranks = np.argsort(np.argsort(total_times, axis=1), axis=1)
    p_win = (ranks == 0).mean(axis=0)
    p_top3 = (ranks < 3).mean(axis=0)
    out = pd.DataFrame({"Driver": drivers.values, "mean_time_s": mean_time, "p_win": p_win, "p_top3": p_top3}).sort_values("mean_time_s").reset_index(drop=True)
    return out, pd.DataFrame()

# ========================= Service =========================
class PodiumService:
    def __init__(self, app_cfg: AppConfig, w_mixed: float = 0.6):
        self.cfg = app_cfg
        self.dual = DualGlobal(app_cfg.model, app_cfg.quant, w_mixed=w_mixed)

    def _train_dual(self, feat_clean: pd.DataFrame, feat_mixed: pd.DataFrame) -> Dict[str, Any]:
        mA = feat_clean["pace_median_s"].notna()
        XA = feat_clean.loc[mA].drop(columns=["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"], errors='ignore')
        yA = feat_clean.loc[mA, "pace_median_s"]
        gA = feat_clean.loc[mA, "race_id"].values
        dA = feat_clean.loc[mA, "Driver"]
        mB = feat_mixed["pace_median_s"].notna()
        XB = feat_mixed.loc[mB].drop(columns=["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"], errors='ignore')
        yB = feat_mixed.loc[mB, "pace_median_s"]
        gB = feat_mixed.loc[mB, "race_id"].values
        dB = feat_mixed.loc[mB, "Driver"]
        return self.dual.fit(XA, yA, gA, dA, XB, yB, gB, dB)

    def predict_podium(self, gp_track_key: str, date: str,
                        seasons_for_training: Tuple[int, ...] = (2022, 2023, 2024),
                        laps: int = 60, pit_loss_s: float = 22.0,
                        pit_count_mean: float = 2.0, sc_rate: float = 0.35,
                        team_order_prob: float = 0.10,
                        weather: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        laps_all = load_sessions(gp_track_key, seasons_for_training)
        if laps_all.empty:
            raise ValueError("No sessions loaded. Check track key/seasons.")
        laps_clean = clean_laps(laps_all, mode="clean")
        laps_mixed = clean_laps(laps_all, mode="mixed")
        drA = build_driver_race_table(laps_clean)
        drB = build_driver_race_table(laps_mixed)
        featA = add_quali_features(drA, laps_clean)
        featB = add_quali_features(drB, laps_mixed)
        # attach weather (constant per race) if provided
        if weather is not None:
            for k, v in weather.items():
                featA[k] = v
                featB[k] = v
        featA = finalize_features(featA)
        featB = finalize_features(featB)
        _ = self._train_dual(featA, featB)
        maskA = featA["circuit_id"].astype(str).str.contains(gp_track_key, case=False, na=False)
        maskB = featB["circuit_id"].astype(str).str.contains(gp_track_key, case=False, na=False)
        XA = featA.loc[maskA].drop(columns=["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"], errors='ignore')
        XB = featB.loc[maskB].drop(columns=["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"], errors='ignore')
        drivers = featA.loc[maskA, "Driver"].reset_index(drop=True)
        if len(XA) == 0 or len(XB) == 0:
            raise ValueError("No inference rows for target GP.")
        q_blend = self.dual.predict_quantiles_blended(XA, XB)
        base_pace = q_blend.get(0.5, self.dual.predict_blended(XA, XB))
        sigma = self.dual.sigma_combined(drivers, q_blend)
        ranking, _ = monte_carlo_podium(drivers, base_pace, sigma, laps, pit_loss_s, pit_count_mean, sc_rate, team_order_prob, n_runs=self.cfg.sim.n_runs, seed=self.cfg.sim.rng_seed)
        return ranking

# ========================= UI — Sidebar Controls =========================
st.sidebar.header("Settings")
col1, col2 = st.sidebar.columns(2)
with col1:
    gp = st.text_input("GP (track key)", value="Monaco")
with col2:
    date = st.text_input("Date (YYYY-MM-DD)", value="2025-05-25")
seasons_str = st.sidebar.text_input("Training seasons (comma)", value="2022,2023,2024")
seasons_tuple = tuple(int(s.strip()) for s in seasons_str.split(",") if s.strip())

st.sidebar.subheader("Race Params")
laps = st.sidebar.number_input("Laps", min_value=1, max_value=1000, value=60)
pit_loss_s = st.sidebar.number_input("Pit loss (s)", min_value=0.0, max_value=60.0, value=22.0, step=0.5)
pit_count_mean = st.sidebar.number_input("Mean pit stops", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
sc_rate = st.sidebar.slider("Safety Car probability", 0.0, 1.0, 0.35, 0.01)
team_order_prob = st.sidebar.slider("Team-order probability", 0.0, 1.0, 0.10, 0.01)

st.sidebar.subheader("Model / Simulation")
# --- Weather controls ---
st.sidebar.subheader("Weather")
wx_mode = st.sidebar.selectbox("Weather source", ["None", "Forecast (Open-Meteo)", "Past 3 days avg (Open-Meteo)"])
lat_in = st.sidebar.text_input("Latitude (optional)", value="")
lon_in = st.sidebar.text_input("Longitude (optional)", value="")

w_mixed = st.sidebar.slider("Blend weight (mixed laps)", 0.0, 1.0, 0.6, 0.05)
n_runs = st.sidebar.number_input("Monte Carlo runs", min_value=100, max_value=10000, value=1500, step=100)

run_btn = st.sidebar.button("Predict podium", type="primary")

# ========================= Run & Display =========================
if run_btn:
    with st.spinner("Loading FastF1 sessions and training models…"):
        try:
            APP_CONFIG.sim.n_runs = int(n_runs)
            svc = PodiumService(APP_CONFIG, w_mixed=float(w_mixed))
            # Weather fetch (optional)
            wx = None
            if wx_mode != "None":
                try:
                    lat_val = float(lat_in) if lat_in.strip() else None
                    lon_val = float(lon_in) if lon_in.strip() else None
                except ValueError:
                    lat_val, lon_val = None, None
                la, lo = _coalesce_coords(gp, lat_val, lon_val)
                if la is None or lo is None:
                    st.warning("No coordinates found. Provide lat/lon or choose a known GP key.")
                else:
                    mode_flag = "forecast" if "Forecast" in wx_mode else "past"
                    try:
                        wx = fetch_weather_open_meteo(la, lo, date, mode=mode_flag)
                    except Exception as we:
                        st.warning(f"Weather fetch failed: {we}")
                        wx = None

            ranking = svc.predict_podium(
                gp_track_key=gp,
                date=date,
                seasons_for_training=seasons_tuple,
                laps=int(laps),
                pit_loss_s=float(pit_loss_s),
                pit_count_mean=float(pit_count_mean),
                sc_rate=float(sc_rate),
                team_order_prob=float(team_order_prob),
                weather=wx,
                gp_track_key=gp,
                date=date,
                seasons_for_training=seasons_tuple,
                laps=int(laps),
                pit_loss_s=float(pit_loss_s),
                pit_count_mean=float(pit_count_mean),
                sc_rate=float(sc_rate),
                team_order_prob=float(team_order_prob),
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Weather summary card
    if wx_mode != "None" and wx is not None:
        st.subheader("Weather (used in features)")
        cwx1, cwx2 = st.columns(2)
        with cwx1:
            st.metric("Temperature", f"{wx['temp_c']:.1f} °C")
        with cwx2:
            st.metric("Rain probability", f"{wx['rain_prob']*100:.0f}%")

    st.subheader("Predicted Ranking")
    fmt = ranking.copy()
    fmt["mean_time_s"] = fmt["mean_time_s"].map(lambda x: f"{x:,.2f}")
    fmt["p_win"] = (fmt["p_win"] * 100).map(lambda x: f"{x:.1f}%")
    fmt["p_top3"] = (fmt["p_top3"] * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(fmt, hide_index=True, use_container_width=True)

    podium = ranking.head(3).reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, col in enumerate([c1, c2, c3]):
        if i < len(podium):
            row = podium.iloc[i]
            with col:
                st.metric(f"{medals[i]} P{i+1}", row["Driver"],
                          help=f"Win {row['p_win']*100:.1f}% · Top3 {row['p_top3']*100:.1f}%")

    st.subheader("Win probability")
    st.bar_chart(ranking.set_index("Driver")["p_win"])

    st.caption("Note: No track meta or weather API attached yet — pit/sc params are user-specified. Quantile + OOF residuals drive uncertainty.")
else:
    st.info("👈 사이드바에서 GP와 파라미터를 설정하고 **Predict podium**을 눌러주세요.")

