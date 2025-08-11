"""
F1 Podium Predictor — Streamlit App (Dual Global clean+mixed + LightGBM + MC)
- FP1/FP2/FP3 롱런 피처 포함
- FastF1로 과거 퀄리 결과 자동 채우기(옵션)
- 수동 엔트리/퀄리 붙여넣기/업로드(옵션)
- 메타테이블 없어도 실행 가능

실행:
    streamlit run app.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import os, io
import datetime as dt
import requests
import numpy as np
import pandas as pd
import fastf1
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
# LightGBM is optional; fall back to scikit-learn if missing
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor as SkGBR
    import warnings
    warnings.filterwarnings("ignore")
import streamlit as st

# ========================= Page Setup =========================
st.set_page_config(page_title="F1 Podium Predictor", layout="wide")
st.title("🏁 F1 Podium Predictor")
st.caption("Dual Global (clean + mixed) · FP features · LightGBM + Quantile · Monte Carlo simulation · Weather-ready")

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
    cv: CVConfig = field(default_factory=CVConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantileConfig = field(default_factory=QuantileConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)

APP_CONFIG = AppConfig()

# ========================= Helpers =========================
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

# ensure cache dir exists to avoid FastF1 NotADirectoryError
os.makedirs("f1_cache", exist_ok=True)
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

# ========================= Data Loading (FP+Q+R) =========================
@st.cache_data(show_spinner=True)
def load_sessions(track_key: str, seasons: Tuple[int, ...]) -> pd.DataFrame:
    recs = []
    for season in seasons:
        rnd = _find_round_in_schedule(season, track_key)
        # include FP1/FP2/FP3 + Q + R
        for sess in ("FP1", "FP2", "FP3", "Q", "R"):
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

# ========================= Cleaning =========================
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

def _max_consecutive_run_length(df: pd.DataFrame) -> int:
    if df.empty or "LapNumber" not in df.columns:
        return 0
    s = df.sort_values("LapNumber")["LapNumber"].to_numpy()
    if len(s) == 0:
        return 0
    max_len = cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 1
    return int(max_len)

def add_practice_features(df_driver_race: pd.DataFrame, df_laps_all: pd.DataFrame) -> pd.DataFrame:
    """FP1/2/3에서 드라이버-레이스 기준:
       - fp_median_s: 세션별 LapTime_s 중앙값들의 중앙
       - fp_std_s:    세션별 LapTime_s 표준편차의 중앙
       - fp_stint_len_max: 연속 주행 최장 랩 수의 최대
    """
    df_fp = df_laps_all[df_laps_all["session"].isin(["FP1", "FP2", "FP3"])].copy()
    if df_fp.empty:
        return df_driver_race
    if "LapTime_s" not in df_fp.columns and "LapTime" in df_fp.columns:
        df_fp["LapTime_s"] = df_fp["LapTime"].dt.total_seconds()

    grp_keys = ["season", "round", "circuit_id", "Driver", "session"]
    per_sess = df_fp.groupby(grp_keys).agg(
        fp_med_s=("LapTime_s", "median"),
        fp_std_s=("LapTime_s", "std"),
        _rows=("LapNumber", "count"),
    ).reset_index()

    stint_rows = []
    for (season, rnd, circ, drv, sess), g in df_fp.groupby(grp_keys, dropna=False):
        stint_rows.append({
            "season": season, "round": rnd, "circuit_id": circ, "Driver": drv, "session": sess,
            "stint_len_max": _max_consecutive_run_length(g)
        })
    per_sess_stint = pd.DataFrame(stint_rows)
    per_sess = per_sess.merge(per_sess_stint, on=grp_keys, how="left")

    grp_keys_race = ["season", "round", "circuit_id", "Driver"]
    per_race = per_sess.groupby(grp_keys_race).agg(
        fp_median_s=("fp_med_s", "median"),
        fp_std_s=("fp_std_s", "median"),
        fp_stint_len_max=("stint_len_max", "max"),
    ).reset_index()

    return df_driver_race.merge(per_race, on=grp_keys_race, how="left")

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "gap_to_pole_s", "best_q_s",
        "sector_sum_s", "s1_mean_s", "s2_mean_s", "s3_mean_s",
        # FP features
        "fp_median_s", "fp_std_s", "fp_stint_len_max",
        # weather (optional)
        "temp_c", "rain_prob",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep + ["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"]]

# ========================= Models (LightGBM + Quantile) =========================
class GlobalRegressor:
    """Global regressor with optional quantile heads and OOF residual stats.
    - Uses LightGBM when available; otherwise falls back to sklearn GradientBoostingRegressor.
    """
    def __init__(self, cfg: ModelConfig, qcfg: QuantileConfig):
        self.cfg = cfg
        self.qcfg = qcfg
        if HAS_LGB:
            self.model = lgb.LGBMRegressor(
                objective=cfg.objective,
                learning_rate=cfg.learning_rate,
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
        else:
            # sklearn fallback (no leaves/colsample params)
            self.model = SkGBR(
                loss="squared_error",
                learning_rate=self.cfg.learning_rate,
                n_estimators=self.cfg.n_estimators,
                max_depth=None if self.cfg.max_depth == -1 else self.cfg.max_depth,
                random_state=42,
            )
        self.imputer = SimpleImputer(strategy="median")
        self.q_models: Dict[float, Any] = {}
        self.resid_std_global_: float = 0.25
        self.resid_std_by_driver_: Dict[str, float] = {}

    def _fit_quantiles(self, Xtr, ytr):
        self.q_models = {}
        if not self.qcfg.use_quantile:
            return
        for q in self.qcfg.quantiles:
            if HAS_LGB:
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
            else:
                qm = SkGBR(
                    loss='quantile',
                    alpha=q,
                    learning_rate=self.cfg.learning_rate,
                    n_estimators=self.cfg.n_estimators,
                    max_depth=None if self.cfg.max_depth == -1 else self.cfg.max_depth,
                    random_state=42,
                )
            qm.fit(Xtr, ytr)
            self.q_models[q] = qm

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
            drivers: Optional[pd.Series] = None) -> Dict[str, Any]:
        X_imp = self.imputer.fit_transform(X)
        # Adjust CV folds to available groups; fallback if too few
        uniq_groups = np.unique(groups)
        n_groups = len(uniq_groups)
        n_splits = min(APP_CONFIG.cv.n_splits, n_groups) if n_groups > 0 else 0
        if n_splits >= 2:
            gkf = GroupKFold(n_splits=n_splits)
            oof = np.zeros_like(y, dtype=float)
            maes = []
            for tr, te in gkf.split(X_imp, y, groups):
                self.model.fit(X_imp[tr], y.iloc[tr])
                pred = self.model.predict(X_imp[te])
                oof[te] = pred
                maes.append(mean_absolute_error(y.iloc[te], pred))
            self.cv_score_ = float(np.mean(maes))
            resid = y.values - oof
        else:
            # Not enough groups for GroupKFold; fit once and use in-sample residuals
            self.model.fit(X_imp, y)
            pred_all = self.model.predict(X_imp)
            self.cv_score_ = float(mean_absolute_error(y, pred_all))
            resid = y.values - pred_all
            # Fit quantiles on the same data in this fallback
            self._fit_quantiles(X_imp, y)
            self.resid_std_global_ = float(np.std(resid)) if len(resid) else 0.25
            if drivers is not None:
                df_res = pd.DataFrame({"Driver": drivers.values, "resid": resid})
                self.resid_std_by_driver_ = df_res.groupby("Driver")["resid"].std().dropna().to_dict()
            return {"cv_mae": self.cv_score_, "resid_std": self.resid_std_global_}
        # After CV, fit on all data and train quantile heads
        self.resid_std_global_ = float(np.std(resid)) if len(resid) else 0.25
        if drivers is not None:
            df_res = pd.DataFrame({"Driver": drivers.values, "resid": resid})
            self.resid_std_by_driver_ = df_res.groupby("Driver")["resid"].std().dropna().to_dict()
        self.model.fit(X_imp, y)
        self._fit_quantiles(X_imp, y)
        return {"cv_mae": self.resid_std_global_, "resid_std": self.resid_std_global_}

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
        return {k: self.w_mixed * qb[k] + (1 - self.w_mixed) * qa[k] for k in keys}

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

# ========================= Auto-fill Entries from FastF1 =========================
def autofill_entries_from_fastf1(season: int, round_number: int, track_key_fallback: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        sess = fastf1.get_session(season, round_number if round_number != -1 else (track_key_fallback or ""), "Q")
        sess.load()
        res = sess.results
        if res is None or res.empty:
            return None
        df = res.copy()
        driver_col = "Abbreviation" if "Abbreviation" in df.columns else ("DriverNumber" if "DriverNumber" in df.columns else None)
        if driver_col is None:
            return None
        out = pd.DataFrame()
        out["Driver"] = df[driver_col].astype(str)
        if "TeamName" in df.columns:
            out["Team"] = df["TeamName"].astype(str)
        elif "Team" in df.columns:
            out["Team"] = df["Team"].astype(str)
        else:
            out["Team"] = ""

        def _td_to_s(x):
            try:
                return x.total_seconds()
            except Exception:
                try:
                    return pd.to_timedelta(x).total_seconds()
                except Exception:
                    return np.nan

        qs = []
        for qcol in ["Q1", "Q2", "Q3"]:
            if qcol in df.columns:
                qs.append(df[qcol].apply(_td_to_s))
        if qs:
            qmin = pd.concat(qs, axis=1).min(axis=1)
        else:
            laps_q = sess.laps.copy()
            laps_q["LapTime_s"] = laps_q["LapTime"].dt.total_seconds()
            qmin = laps_q.groupby("Driver")["LapTime_s"].min().reindex(out["Driver"]).values
        out["QualiTime"] = qmin
        out = out.dropna(subset=["Driver"]).reset_index(drop=True)
        return out
    except Exception:
        return None

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
                       weather: Optional[Dict[str, float]] = None,
                       entries: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        # Load + Clean
        laps_all = load_sessions(gp_track_key, seasons_for_training)
        if laps_all.empty:
            raise ValueError("No sessions loaded. Check track key/seasons.")
        laps_clean = clean_laps(laps_all, mode="clean")
        laps_mixed = clean_laps(laps_all, mode="mixed")

        # Features
        drA = build_driver_race_table(laps_clean)
        drB = build_driver_race_table(laps_mixed)
        featA = add_quali_features(drA, laps_clean)
        featB = add_quali_features(drB, laps_mixed)
        featA = add_practice_features(featA, laps_clean)
        featB = add_practice_features(featB, laps_mixed)
        if weather is not None:
            for k, v in weather.items():
                featA[k] = v
                featB[k] = v
        featA = finalize_features(featA)
        featB = finalize_features(featB)

        # Target race = latest season/round for this GP
        maskA_all = featA["circuit_id"].astype(str).str.contains(gp_track_key, case=False, na=False)
        if not maskA_all.any():
            raise ValueError("No historical rows for target GP. Try adding more seasons.")
        latest_season = int(featA.loc[maskA_all, "season"].max())
        latest_round = int(featA.loc[(maskA_all) & (featA["season"] == latest_season), "round"].max())

        infA = featA[(featA["season"] == latest_season) & (featA["round"] == latest_round)].copy()
        infB = featB[(featB["season"] == latest_season) & (featB["round"] == latest_round)].copy()
        if len(infA) == 0 or len(infB) == 0:
            raise ValueError("No inference rows for latest target race (missing FP/Q/R).")

        # Entries override
        if entries is not None and len(entries) > 0:
            ent = entries.copy()
            ent.columns = [c.strip() for c in ent.columns]
            colmap = {c.lower(): c for c in ent.columns}
            if "driver" not in colmap:
                raise ValueError("Entries must include a 'Driver' column")
            ent.rename(columns={colmap["driver"]: "Driver"}, inplace=True)
            if "team" in colmap: ent.rename(columns={colmap["team"]: "Team"}, inplace=True)
            if "qualitime (s)" in colmap: ent.rename(columns={colmap["qualitime (s)"]: "QualiTime"}, inplace=True)
            if "qualitime" in colmap and "QualiTime" not in ent.columns:
                ent.rename(columns={colmap["qualitime"]: "QualiTime"}, inplace=True)
            ent = ent[[c for c in ["Driver","Team","QualiTime"] if c in ent.columns]]

            listed = set(ent["Driver"].astype(str).unique())
            infA = infA[infA["Driver"].astype(str).isin(listed)].copy()
            infB = infB[infB["Driver"].astype(str).isin(listed)].copy()
            if len(infA) == 0:
                raise ValueError("None of the provided drivers match the target race. Check driver codes (e.g., VER, NOR).")

            if "QualiTime" in ent.columns and ent["QualiTime"].notna().any():
                def _to_seconds(x):
                    try:
                        return float(x)
                    except Exception:
                        s = str(x).strip()
                        if ":" in s:
                            mm, ss = s.split(":", 1)
                            return float(mm) * 60 + float(ss)
                        return np.nan
                ent["best_q_s"] = ent["QualiTime"].apply(_to_seconds)
                if ent["best_q_s"].notna().any():
                    pole = float(np.nanmin(ent["best_q_s"].values))
                    ent["gap_to_pole_s"] = ent["best_q_s"] - pole
                    for tag in ("A","B"):
                        df = infA if tag=="A" else infB
                        df = df.drop(columns=["best_q_s","gap_to_pole_s"], errors="ignore")
                        df = df.merge(ent[["Driver","best_q_s","gap_to_pole_s"]], on="Driver", how="left")
                        if tag=="A": infA = df
                        else: infB = df

        XA = infA.drop(columns=["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"], errors='ignore')
        XB = infB.drop(columns=["pace_median_s", "season", "round", "circuit_id", "Driver", "race_id"], errors='ignore')
        drivers = infA["Driver"].reset_index(drop=True)

        # Train on everything EXCEPT the target race (no leakage)
        trA = featA[~((featA["season"] == latest_season) & (featA["round"] == latest_round))]
        trB = featB[~((featB["season"] == latest_season) & (featB["round"] == latest_round))]
        _ = self._train_dual(trA, trB)

        if len(XA) == 0 or len(XB) == 0:
            raise ValueError("No inference rows for target GP.")
        q_blend = self.dual.predict_quantiles_blended(XA, XB)
        base_pace = q_blend.get(0.5, self.dual.predict_blended(XA, XB))
        sigma = self.dual.sigma_combined(drivers, q_blend)
        ranking, _ = monte_carlo_podium(drivers, base_pace, sigma, laps, pit_loss_s, pit_count_mean, sc_rate, team_order_prob,
                                        n_runs=self.cfg.sim.n_runs, seed=self.cfg.sim.rng_seed)
        return ranking

# ========================= UI — Sidebar Controls =========================
st.sidebar.header("Settings")

KNOWN_GPS = sorted(list(TRACK_COORDS.keys()) + [
    "Australia", "Spain", "Canada", "Austria", "Hungary", "Belgium",
    "Netherlands", "Singapore", "USA", "Mexico", "Brazil", "Abu Dhabi"
])

gp_mode = st.sidebar.radio("GP input mode", ["Pick from list", "Free text"], horizontal=True)
if gp_mode == "Pick from list":
    gp = st.sidebar.selectbox("Grand Prix", KNOWN_GPS, index=KNOWN_GPS.index("Monaco") if "Monaco" in KNOWN_GPS else 0)
else:
    gp = st.sidebar.text_input("GP (track key)", value="Monaco")

col1, col2 = st.sidebar.columns(2)
with col1:
    date = st.text_input("Date (YYYY-MM-DD)", value="2025-05-25")
with col2:
    season_choices = list(range(2018, dt.date.today().year+1))
    default_seasons = [2022, 2023, 2024]
seasons_selected = st.sidebar.multiselect("Training seasons", options=season_choices, default=default_seasons)
seasons_tuple = tuple(sorted(set(int(s) for s in seasons_selected)))
if len(seasons_tuple) < 2:
    st.sidebar.info("Tip: Add more seasons for stabler CV")

with st.sidebar.expander("Advanced settings", expanded=False):
    st.markdown("**Race Params**")
    laps = st.number_input("Laps", min_value=1, max_value=1000, value=60, key="laps")
    pit_loss_s = st.number_input("Pit loss (s)", min_value=0.0, max_value=60.0, value=22.0, step=0.5, key="pitloss")
    pit_count_mean = st.number_input("Mean pit stops", min_value=0.0, max_value=6.0, value=2.0, step=0.1, key="pitmean")
    sc_rate = st.slider("Safety Car probability", 0.0, 1.0, 0.35, 0.01, key="sc")
    team_order_prob = st.slider("Team-order probability", 0.0, 1.0, 0.10, 0.01, key="teamorder")

    st.markdown("**Model / Simulation**")
    w_mixed = st.slider("Blend weight (mixed laps)", 0.0, 1.0, 0.6, 0.05, key="wmixed")
    n_runs = st.number_input("Monte Carlo runs", min_value=100, max_value=10000, value=1500, step=100, key="nruns")
    rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1, key="seed")

    st.markdown("**Weather**")
    wx_mode = st.selectbox("Weather source", ["None", "Forecast (Open-Meteo)", "Past 3 days avg (Open-Meteo)"], key="wxmode")
    lat_in = st.text_input("Latitude (optional)", value="", key="lat")
    lon_in = st.text_input("Longitude (optional)", value="", key="lon")

# Entries / Quali 입력
with st.sidebar.expander("Entries / Qualifying (optional)", expanded=False):
    mode_ent = st.radio("Provide entries?", ["None", "Paste CSV", "Upload CSV"], horizontal=True)
    entries_df = None
    if mode_ent == "Paste CSV":
        sample = "Driver,Team,QualiTime\nVER,Red Bull,86.204\nNOR,McLaren,86.269\nPIA,McLaren,86.375\n"
        txt = st.text_area("Paste CSV here", value=sample, height=160)
        if txt.strip():
            try:
                entries_df = pd.read_csv(io.StringIO(txt))
                st.caption(f"Parsed {len(entries_df)} rows")
            except Exception as e:
                st.warning(f"CSV parse failed: {e}")
    elif mode_ent == "Upload CSV":
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f is not None:
            try:
                entries_df = pd.read_csv(f)
                st.caption(f"Parsed {len(entries_df)} rows")
            except Exception as e:
                st.warning(f"CSV parse failed: {e}")

# Auto-fill
with st.sidebar.expander("Auto-fill (FastF1)", expanded=False):
    use_autofill = st.checkbox("Use FastF1 qualifying entries (past races only)", value=False)

# Cache tools
colc1, colc2 = st.sidebar.columns([1,1])
with colc1:
    run_btn = st.button("Predict podium", type="primary")
with colc2:
    if st.button("Clear FastF1 cache"):
        import shutil
        try:
            shutil.rmtree("f1_cache")
            os.makedirs("f1_cache", exist_ok=True)
            st.sidebar.success("Cache cleared")
        except Exception as e:
            st.sidebar.warning(f"Cache clear failed: {e}")

# ========================= Run & Display =========================
if run_btn:
    with st.status("Preparing…", expanded=False) as status:
        status.update(label="Loading sessions", state="running")
        try:
            APP_CONFIG.sim.n_runs = int(n_runs)
            APP_CONFIG.sim.rng_seed = int(rng_seed)
            svc = PodiumService(APP_CONFIG, w_mixed=float(w_mixed))

            # Weather
            wx = None
            if st.session_state.get("wxmode", "None") != "None":
                try:
                    lat_val = float(lat_in) if str(lat_in).strip() else None
                    lon_val = float(lon_in) if str(lon_in).strip() else None
                except ValueError:
                    lat_val, lon_val = None, None
                la, lo = _coalesce_coords(gp, lat_val, lon_val)
                if la is None or lo is None:
                    st.warning("No coordinates found. Provide lat/lon or choose a known GP key.")
                else:
                    mode_flag = "forecast" if "Forecast" in st.session_state["wxmode"] else "past"
                    try:
                        wx = fetch_weather_open_meteo(la, lo, date, mode=mode_flag)
                    except Exception as we:
                        st.warning(f"Weather fetch failed: {we}")
                        wx = None

            # Auto-fill entries if requested (uses latest selected season)
            if use_autofill and (seasons_tuple):
                try:
                    latest_season_candidate = max(seasons_tuple)
                    rnd = _find_round_in_schedule(latest_season_candidate, gp)
                    auto_df = autofill_entries_from_fastf1(latest_season_candidate, rnd, track_key_fallback=gp)
                    if auto_df is not None and not auto_df.empty:
                        entries_df = auto_df
                        st.sidebar.success(f"Auto-filled {len(entries_df)} entries from FastF1")
                    else:
                        st.sidebar.warning("Could not auto-fill entries from FastF1 for this GP/season.")
                except Exception:
                    st.sidebar.warning("Auto-fill failed.")

            ranking = svc.predict_podium(
                gp_track_key=gp,
                date=date,
                seasons_for_training=seasons_tuple if seasons_tuple else (2022, 2023, 2024),
                laps=int(st.session_state.get("laps", 60)),
                pit_loss_s=float(st.session_state.get("pitloss", 22.0)),
                pit_count_mean=float(st.session_state.get("pitmean", 2.0)),
                sc_rate=float(st.session_state.get("sc", 0.35)),
                team_order_prob=float(st.session_state.get("teamorder", 0.10)),
                weather=wx,
                entries=entries_df,
            )
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Error: {e}")
            st.stop()

    # Weather summary
    if st.session_state.get("wxmode", "None") != "None" and wx is not None:
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

    cdl1, cdl2 = st.columns([1,1])
    with cdl1:
        st.download_button("Download CSV", data=ranking.to_csv(index=False).encode("utf-8"),
                           file_name=f"podium_{gp.replace(' ','_')}.csv", mime="text/csv")
    with cdl2:
        st.download_button("Download JSON", data=ranking.to_json(orient="records").encode("utf-8"),
                           file_name=f"podium_{gp.replace(' ','_')}.json", mime="application/json")

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

    st.caption("Note: Track meta CSV not attached yet — pit/SC params are user-specified. Weather uses Open-Meteo; uncertainty from quantiles + OOF residuals.")
else:
    st.info("👈 사이드바에서 GP와 파라미터를 설정하고 **Predict podium**을 눌러주세요.")
