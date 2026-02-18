# ============================================
# NHL_Simulation.py  (SANITIZED DEMO VERSION)
# --------------------------------------------
# Runs with ONLY these required files:
#   data/NHL_5v5_2025_26.csv
#   data/NHL_5v4_2025_26.csv
#   data/NHL_4v5_2025_26.csv
#
# Usage:
#   python NHL_Simulation.py BOS MTL
#   python NHL_Simulation.py BOS MTL --sims 50000
#
# Notes:
# - Any “production” modules (rest/goalies/pace/home-ice/injuries) are OPTIONAL.
# - If those CSVs are absent, the sim prints ONE demo line and continues with neutral defaults.
# - CSV column names are auto-detected (handles common variants).
# ============================================

import os
import sys
import math
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PATHS = {
    # REQUIRED
    "5v5": os.path.join(DATA_DIR, "NHL_5v5_2025_26.csv"),
    "5v4": os.path.join(DATA_DIR, "NHL_5v4_2025_26.csv"),
    "4v5": os.path.join(DATA_DIR, "NHL_4v5_2025_26.csv"),

    # OPTIONAL (demo-safe)
    "rest": os.path.join(DATA_DIR, "rest_factors.csv"),
    "goalie": os.path.join(DATA_DIR, "goalie_multipliers.csv"),
    "pace": os.path.join(DATA_DIR, "pace_factors.csv"),
    "home_ice": os.path.join(DATA_DIR, "home_ice.csv"),
    "injuries": os.path.join(DATA_DIR, "injuries.csv"),
}


# -----------------------------
# Demo-safe IO
# -----------------------------
# Tracks if ANY optional module is missing or failed to load
OPTIONAL_MISSING = False
_DEMO_LINE_PRINTED = False

def _warn(msg: str) -> None:
    # Do not spam warnings; just record that optional modules are disabled.
    global OPTIONAL_MISSING
    OPTIONAL_MISSING = True

def _print_demo_line_once() -> None:
    global _DEMO_LINE_PRINTED
    if not _DEMO_LINE_PRINTED:
        print("[demo] Running demo configuration (optional production modules disabled).")
        _DEMO_LINE_PRINTED = True

def require_csv(path: str, what: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {what}\n"
            f"Expected: {path}\n"
            f"Repo demo requires ONLY the three core CSVs in /data."
        )
    return pd.read_csv(path)

def optional_csv(path: str, default_df: pd.DataFrame, what: str) -> pd.DataFrame:
    if not os.path.exists(path):
        _warn(f"{what} not found ({os.path.basename(path)}). Using neutral defaults.")
        return default_df
    try:
        return pd.read_csv(path)
    except Exception as e:
        _warn(f"Failed reading {what} ({os.path.basename(path)}): {type(e).__name__}: {e}. Using defaults.")
        return default_df


# -----------------------------
# Column detection helpers
# -----------------------------
def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # try fuzzy contains
    for cand in candidates:
        cand_l = cand.lower()
        for c in df.columns:
            if cand_l in c.lower():
                return c
    return None

def _require_col(df: pd.DataFrame, candidates, label: str) -> str:
    col = _find_col(df, candidates)
    if col is None:
        raise ValueError(
            f"Could not find a '{label}' column in CSV.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return col

def load_team_rates(path_key: str, what: str) -> Dict[str, Dict[str, float]]:
    """
    Loads team rates from a core CSV and returns:
      rates[TEAM] = {"xGF60": ..., "xGA60": ...}
    Auto-detects common column names.
    """
    df = require_csv(PATHS[path_key], what)

    team_col = _require_col(df, ["team", "Team", "Tm", "team_abbrev", "abbrev"], "team")
    xgf_col = _require_col(df, ["xGF/60", "xGF60", "xGF_60", "xgf60", "xgf_per60", "xgf"], "xGF/60")
    xga_col = _require_col(df, ["xGA/60", "xGA60", "xGA_60", "xga60", "xga_per60", "xga"], "xGA/60")

    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        t = str(r[team_col]).strip().upper()
        if not t:
            continue
        out[t] = {
            "xGF60": float(r[xgf_col]),
            "xGA60": float(r[xga_col]),
        }
    return out


# -----------------------------
# Optional modifier lookups
# -----------------------------
def build_lookup(df: pd.DataFrame, key_col: str, val_col: str) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    if key_col not in df.columns or val_col not in df.columns:
        return {}
    out = {}
    for _, r in df[[key_col, val_col]].dropna().iterrows():
        out[str(r[key_col]).strip().upper()] = float(r[val_col])
    return out

def load_optional_modules() -> Dict[str, Dict[str, float]]:
    # Minimal schema defaults
    rest_df = optional_csv(PATHS["rest"], pd.DataFrame(columns=["team", "rest_modifier"]), "rest_factors")
    goalie_df = optional_csv(PATHS["goalie"], pd.DataFrame(columns=["team", "goalie_multiplier"]), "goalie_multipliers")
    pace_df = optional_csv(PATHS["pace"], pd.DataFrame(columns=["team", "pace_factor"]), "pace_factors")
    home_df = optional_csv(PATHS["home_ice"], pd.DataFrame(columns=["team", "home_ice_multiplier"]), "home_ice")
    inj_df = optional_csv(PATHS["injuries"], pd.DataFrame(columns=["team", "injury_off_mult", "injury_def_mult"]), "injuries")

    rest_lu = build_lookup(rest_df, "team", "rest_modifier")
    goalie_lu = build_lookup(goalie_df, "team", "goalie_multiplier")
    pace_lu = build_lookup(pace_df, "team", "pace_factor")
    home_lu = build_lookup(home_df, "team", "home_ice_multiplier")
    inj_off_lu = build_lookup(inj_df, "team", "injury_off_mult")
    inj_def_lu = build_lookup(inj_df, "team", "injury_def_mult")

    return {
        "rest": rest_lu,
        "goalie": goalie_lu,
        "pace": pace_lu,
        "home": home_lu,
        "inj_off": inj_off_lu,
        "inj_def": inj_def_lu,
    }

def team_mods(team: str, is_home: bool, modules: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    t = team.strip().upper()
    return {
        "rest": modules["rest"].get(t, 1.0),
        "goalie": modules["goalie"].get(t, 1.0),
        "pace": modules["pace"].get(t, 1.0),
        "home": modules["home"].get(t, 1.0) if is_home else 1.0,
        "inj_off": modules["inj_off"].get(t, 1.0),
        "inj_def": modules["inj_def"].get(t, 1.0),
    }


# -----------------------------
# Core math
# -----------------------------
def american_from_prob(p: float) -> int:
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def poisson_rng(lam: float, size: int) -> np.ndarray:
    lam = max(lam, 0.0)
    return np.random.poisson(lam=lam, size=size)

def blend_attack_defense(off_xgf60: float, opp_xga60: float) -> float:
    """
    Simple neutral blend for demo: arithmetic mean of offense and opponent defense allowance.
    (Keeps behavior stable and easy to explain in a public repo.)
    """
    return 0.5 * (off_xgf60 + opp_xga60)

def expected_goals_5v5(home: str, away: str,
                      rates_5v5: Dict[str, Dict[str, float]],
                      mods: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """
    Compute baseline expected goals at 5v5 over 60 minutes.
    Then apply optional neutral modifiers (rest/goalie/home/injuries).
    Pace affects totals by scaling both teams equally (demo-friendly).
    """
    h = home.upper(); a = away.upper()
    if h not in rates_5v5:
        raise KeyError(f"Home team '{home}' not found in 5v5 CSV.")
    if a not in rates_5v5:
        raise KeyError(f"Away team '{away}' not found in 5v5 CSV.")

    h_off = rates_5v5[h]["xGF60"]
    h_def_allow = rates_5v5[h]["xGA60"]
    a_off = rates_5v5[a]["xGF60"]
    a_def_allow = rates_5v5[a]["xGA60"]

    # Baseline EG/60 using offense vs opponent allowance
    home_mu = blend_attack_defense(h_off, a_def_allow)
    away_mu = blend_attack_defense(a_off, h_def_allow)

    # Optional modifiers
    hm = team_mods(h, True, mods)
    am = team_mods(a, False, mods)

    # Home ice: boost home scoring only
    home_mu *= hm["home"]

    # Rest: apply to offense (simple)
    home_mu *= hm["rest"]
    away_mu *= am["rest"]

    # Injuries: offense and defense (def multiplier applied inversely to opponent scoring)
    home_mu *= hm["inj_off"] * (1.0 / max(am["inj_def"], 1e-9))
    away_mu *= am["inj_off"] * (1.0 / max(hm["inj_def"], 1e-9))

    # Goalie multiplier: for demo we interpret goalie_multiplier as "opponent scoring multiplier"
    # (neutral=1.0 so harmless if absent)
    home_mu *= am["goalie"]
    away_mu *= hm["goalie"]

    # Pace: scale both teams equally (totals-only feel)
    pace_scale = math.sqrt(hm["pace"] * am["pace"])
    home_mu *= pace_scale
    away_mu *= pace_scale

    return home_mu, away_mu

def special_teams_adjustment(home: str, away: str,
                             rates_5v4: Dict[str, Dict[str, float]],
                             rates_4v5: Dict[str, Dict[str, float]],
                             pp_minutes: float,
                             pk_minutes: float) -> Tuple[float, float]:
    """
    Very demo-safe special teams model:
    - Home PP uses home 5v4 offense vs away 4v5 allowance.
    - Away PP uses away 5v4 offense vs home 4v5 allowance.
    Convert per-60 rates into expected goals over pp_minutes.
    """
    h = home.upper(); a = away.upper()
    for dct, name in [(rates_5v4, "5v4"), (rates_4v5, "4v5")]:
        if h not in dct:
            raise KeyError(f"Team '{home}' not found in {name} CSV.")
        if a not in dct:
            raise KeyError(f"Team '{away}' not found in {name} CSV.")

    # Home PP offense
    h_pp_off = rates_5v4[h]["xGF60"]
    # Away PK allowance (as xGA/60 while shorthanded)
    a_pk_allow = rates_4v5[a]["xGA60"]

    # Away PP offense
    a_pp_off = rates_5v4[a]["xGF60"]
    # Home PK allowance
    h_pk_allow = rates_4v5[h]["xGA60"]

    # Blend to get PP scoring rates
    home_pp_rate60 = blend_attack_defense(h_pp_off, a_pk_allow)
    away_pp_rate60 = blend_attack_defense(a_pp_off, h_pk_allow)

    # Expected PP goals (minutes → fraction of hour)
    home_pp_goals = home_pp_rate60 * (pp_minutes / 60.0)
    away_pp_goals = away_pp_rate60 * (pp_minutes / 60.0)

    # Optional: PK minutes symmetric in this toy model (kept for readability)
    _ = pk_minutes  # placeholder

    return home_pp_goals, away_pp_goals


def simulate_matchup(home: str, away: str,
                     rates_5v5: Dict[str, Dict[str, float]],
                     rates_5v4: Dict[str, Dict[str, float]],
                     rates_4v5: Dict[str, Dict[str, float]],
                     sims: int,
                     pp_minutes: float,
                     include_ot: bool,
                     seed: Optional[int]) -> Dict[str, float]:
    if seed is not None:
        np.random.seed(seed)

    modules = load_optional_modules()

    # 5v5 expected goals per 60
    home_mu_60, away_mu_60 = expected_goals_5v5(home, away, rates_5v5, modules)

    # Convert 5v5 to regulation minutes minus PP time for simplicity
    reg_minutes = 60.0
    fivevfive_minutes = max(reg_minutes - pp_minutes, 0.0)

    home_5v5_mu = home_mu_60 * (fivevfive_minutes / 60.0)
    away_5v5_mu = away_mu_60 * (fivevfive_minutes / 60.0)

    # Special teams expected goals (both teams get same PP time in this toy demo)
    home_pp_mu, away_pp_mu = special_teams_adjustment(home, away, rates_5v4, rates_4v5, pp_minutes=pp_minutes, pk_minutes=pp_minutes)

    # Total regulation lambdas
    home_lam = home_5v5_mu + home_pp_mu
    away_lam = away_5v5_mu + away_pp_mu

    # Simulate regulation
    hg = poisson_rng(home_lam, sims)
    ag = poisson_rng(away_lam, sims)

    home_wins = (hg > ag).sum()
    away_wins = (ag > hg).sum()
    ties = sims - home_wins - away_wins

    # OT handling: simple 3v3-ish coinflip weighted by slight edge from regulation mu
    if include_ot and ties > 0:
        p_home_ot = home_mu_60 / max(home_mu_60 + away_mu_60, 1e-9)
        tie_mask = (hg == ag)
        ot_draws = np.random.rand(tie_mask.sum())
        home_ot_wins = (ot_draws < p_home_ot).sum()
        away_ot_wins = tie_mask.sum() - home_ot_wins
        home_wins += home_ot_wins
        away_wins += away_ot_wins

    p_home = home_wins / sims
    p_away = away_wins / sims
    avg_total = float((hg + ag).mean())
    avg_home = float(hg.mean())
    avg_away = float(ag.mean())

    return {
        "p_home": p_home,
        "p_away": p_away,
        "home_ml": american_from_prob(p_home),
        "away_ml": american_from_prob(p_away),
        "home_xg": avg_home,
        "away_xg": avg_away,
        "total_xg": avg_total,
        "home_lam": float(home_lam),
        "away_lam": float(away_lam),
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Sanitized NHL simulation demo (minimal required data).")
    ap.add_argument("home", type=str, help="Home team abbrev, e.g., BOS")
    ap.add_argument("away", type=str, help="Away team abbrev, e.g., MTL")
    ap.add_argument("--sims", type=int, default=50000, help="Number of Monte Carlo sims (default 50000)")
    ap.add_argument("--pp-minutes", type=float, default=10.0, help="Total PP minutes per team (toy demo, default 10.0)")
    ap.add_argument("--no-ot", action="store_true", help="Do not resolve ties with OT (ties count as no-win)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Load required core rates
    rates_5v5 = load_team_rates("5v5", "5v5 team rates (xGF/60, xGA/60)")
    rates_5v4 = load_team_rates("5v4", "5v4 team rates (PP xGF/60, xGA/60)")
    rates_4v5 = load_team_rates("4v5", "4v5 team rates (PK xGF/60, xGA/60)")

    res = simulate_matchup(
        home=args.home,
        away=args.away,
        rates_5v5=rates_5v5,
        rates_5v4=rates_5v4,
        rates_4v5=rates_4v5,
        sims=args.sims,
        pp_minutes=args.pp_minutes,
        include_ot=(not args.no_ot),
        seed=args.seed,
    )

    # Print ONE demo line if optional modules were missing/disabled
    if OPTIONAL_MISSING:
        _print_demo_line_once()

    home = args.home.upper()
    away = args.away.upper()

    print("\n============================================")
    print(f"Matchup: {away} @ {home}")
    print("============================================")
    print(f"Sims: {args.sims:,}")
    print(f"PP minutes (per team, toy): {args.pp_minutes:.1f}")
    print("--------------------------------------------")
    print(f"Win%  {home}: {res['p_home']*100:.2f}%   (fair ML {res['home_ml']:+d})")
    print(f"Win%  {away}: {res['p_away']*100:.2f}%   (fair ML {res['away_ml']:+d})")
    print("--------------------------------------------")
    print(f"Expected goals (avg): {home} {res['home_xg']:.2f}  |  {away} {res['away_xg']:.2f}")
    print(f"Expected total: {res['total_xg']:.2f}")
    print("--------------------------------------------")
    print(f"Reg lambdas: {home} {res['home_lam']:.3f}  |  {away} {res['away_lam']:.3f}")
    print("============================================\n")


if __name__ == "__main__":
    main()
