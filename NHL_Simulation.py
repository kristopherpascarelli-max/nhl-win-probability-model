# ============================================
# NHL Simulation — Portfolio Sample
# No empty-net modeling
# Portable paths + clean loads
# ============================================

import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
from collections import defaultdict
from datetime import date
import argparse

# ============================================
# Paths (portable)
# ============================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PATHS = {
    "goalie_mult":     DATA / "goalie_multipliers.csv",
    "power":           DATA / "team_power.csv",

    "xG_5v5":          DATA / "NHL_5v5_2025_26.csv",
    "xG_5v4":          DATA / "NHL_5v4_2025_26.csv",
    "xG_4v5":          DATA / "NHL_4v5_2025_26.csv",

    "rest_factors":    DATA / "rest_factors.csv",
    "rest_today":      DATA / "team_rest_today.csv",
    "pace":            DATA / "team_pace_factors.csv",

    "injury_tiers":    DATA / "injury_tiers.csv",
    "penalties_taken": DATA / "team_penalties_taken_2025_26.csv",

    "export":          ROOT / "projections.csv",
}

# ============================================
# Config
# ============================================
PACE_EFFECT_STRENGTH = 0.10
GOALIE_SHRINK = 0.70
PP_SCALER = 0.75

HFA_LOGIT = 0.1205
OT_SCALER = 0.65
OT_HOME_EDGE = 0.02

TEAM_POWER_WEIGHT = 0.20

OFF_PEN_PER_TIER = 0.08
DEF_PEN_PER_TIER = 0.04
D_OFF_PEN_PER_TIER = 0.12

# ============================================
# Math helpers
# ============================================
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))

def poisson_pmf(k, lam):
    if lam <= 0 and k == 0:
        return 1.0
    if lam <= 0:
        return 0.0
    return (lam ** k * math.exp(-lam)) / math.factorial(k)

def score_diff_probs(mu_a, mu_h, kmax=12):
    pa = [poisson_pmf(k, mu_a) for k in range(kmax + 1)]
    ph = [poisson_pmf(k, mu_h) for k in range(kmax + 1)]
    diff = defaultdict(float)
    for ha in range(kmax + 1):
        for aa in range(kmax + 1):
            diff[ha - aa] += ph[ha] * pa[aa]
    return diff

def require_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")

# ============================================
# Loaders
# ============================================
def load_xG_csv(path, label):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["xgf", "xgf_per60", "xgf/60", "xgf_per_60"]:
        if c in df.columns:
            df.rename(columns={c: "xgf"}, inplace=True)
            break
    for c in ["xga", "xga_per60", "xga/60", "xga_per_60"]:
        if c in df.columns:
            df.rename(columns={c: "xga"}, inplace=True)
            break

    require_columns(df, ["team", "xgf", "xga"], label)
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    return df[["team", "xgf", "xga"]].copy()

def load_rest_factors():
    df = pd.read_csv(PATHS["rest_factors"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["rest_state", "logit_adj", "xg_mult"], "rest_factors")

    df["rest_state"] = df["rest_state"].astype(str).str.upper().str.strip()
    logit_map = dict(zip(df["rest_state"], pd.to_numeric(df["logit_adj"], errors="coerce")))
    mult_map  = dict(zip(df["rest_state"], pd.to_numeric(df["xg_mult"], errors="coerce")))
    return logit_map, mult_map

def load_team_rest():
    df = pd.read_csv(PATHS["rest_today"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["team", "rest_state"], "team_rest_today")
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["rest_state"] = df["rest_state"].astype(str).str.upper().str.strip()
    return dict(zip(df["team"], df["rest_state"]))

def load_pace_factors():
    df = pd.read_csv(PATHS["pace"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["team", "pace_factor"], "team_pace_factors")
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    pf = pd.to_numeric(df["pace_factor"], errors="coerce")
    return dict(zip(df["team"], pf))

def load_goalie_table():
    df = pd.read_csv(PATHS["goalie_mult"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["team", "role", "multiplier"], "goalie_multipliers")
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["role"] = df["role"].astype(str).str.lower().str.strip()
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
    return df

def load_penalties_taken():
    df = pd.read_csv(PATHS["penalties_taken"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["team", "penalties_taken_pg"], "team_penalties_taken")
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["penalties_taken_pg"] = pd.to_numeric(df["penalties_taken_pg"], errors="coerce").fillna(0.0)
    return dict(zip(df["team"], df["penalties_taken_pg"]))

def load_injury_tiers():
    if not os.path.exists(PATHS["injury_tiers"]):
        return {}, {}
    df = pd.read_csv(PATHS["injury_tiers"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["team", "missing_d_tiers", "missing_f_tiers"], "injury_tiers")
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["missing_d_tiers"] = pd.to_numeric(df["missing_d_tiers"], errors="coerce").fillna(0.0)
    df["missing_f_tiers"] = pd.to_numeric(df["missing_f_tiers"], errors="coerce").fillna(0.0)
    return dict(zip(df["team"], df["missing_d_tiers"])), dict(zip(df["team"], df["missing_f_tiers"]))

def load_team_power():
    df = pd.read_csv(PATHS["power"])
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, ["team", "rating_logit"], "team_power")
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["rating_logit"] = pd.to_numeric(df["rating_logit"], errors="coerce").fillna(0.0)
    return dict(zip(df["team"], df["rating_logit"]))

# ============================================
# Global loads
# ============================================
xG_5v5 = load_xG_csv(PATHS["xG_5v5"], "5v5")
xG_5v4 = load_xG_csv(PATHS["xG_5v4"], "5v4")
xG_4v5 = load_xG_csv(PATHS["xG_4v5"], "4v5")

LEAGUE_MEANS = {
    "5v5": (xG_5v5["xgf"].mean(), xG_5v5["xga"].mean()),
    "5v4": (xG_5v4["xgf"].mean(), xG_5v4["xga"].mean()),
    "4v5": (xG_4v5["xgf"].mean(), xG_4v5["xga"].mean()),
}

REST_LOGIT, REST_XGMULT = load_rest_factors()
TEAM_REST_STATE = load_team_rest()
PACE = load_pace_factors()
GOALIES = load_goalie_table()
PEN_TAKEN = load_penalties_taken()
INJURY_TIERS_D, INJURY_TIERS_F = load_injury_tiers()
TEAM_POWER = load_team_power()

DICT_XG = {
    "xG_5v5":  dict(zip(xG_5v5.team, xG_5v5.xgf)),
    "xGA_5v5": dict(zip(xG_5v5.team, xG_5v5.xga)),
    "xG_5v4":  dict(zip(xG_5v4.team, xG_5v4.xgf)),
    "xGA_4v5": dict(zip(xG_4v5.team, xG_4v5.xga)),
    "xG_4v5":  dict(zip(xG_4v5.team, xG_4v5.xgf)),
    "xGA_5v4": dict(zip(xG_5v4.team, xG_5v4.xga)),
}

# ============================================
# Core simulation (no EN)
# ============================================
def simulate_game(away, home, a_role, h_role):

    lam_home = DICT_XG["xG_5v5"][home]
    lam_away = DICT_XG["xG_5v5"][away]

    # Basic ML expectation
    base_sum = lam_home + lam_away
    p_base = lam_home / max(1e-12, base_sum)

    dlogit = TEAM_POWER_WEIGHT * (TEAM_POWER[home] - TEAM_POWER[away])
    dlogit += HFA_LOGIT

    p_adj = logistic(logit(p_base) + dlogit)

    p_home_power = p_adj
    p_away_power = 1.0 - p_home_power

    # Regulation tie probability (from Poisson)
    d_reg = score_diff_probs(lam_away, lam_home)
    p_tie = d_reg.get(0, 0.0)

    # OT mix
    p_home_ot = 0.5 + OT_HOME_EDGE + (p_home_power - 0.5) * OT_SCALER
    p_home_ot = min(max(p_home_ot, 0.01), 0.99)

    p_home_final = (1.0 - p_tie) * p_home_power + p_tie * p_home_ot

    mu_total = lam_home + lam_away

    return {
        "p_home_final": p_home_final,
        "mu_total": mu_total,
        "p_tie": p_tie
    }

def moneyline_from_prob(p):
    if p > 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("AWAY")
    parser.add_argument("HOME")
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    sums = {"p_home_final": 0.0, "mu_total": 0.0, "p_tie": 0.0}

    for _ in range(args.sims):
        r = simulate_game(args.AWAY.upper(), args.HOME.upper(), "blended", "blended")
        for k in sums:
            sums[k] += r[k]

    avg = {k: sums[k] / args.sims for k in sums}

    p_home = avg["p_home_final"]
    p_away = 1.0 - p_home

    print(f"{args.AWAY} @ {args.HOME}")
    print(f"Home win%: {p_home:.3f}  (ML {moneyline_from_prob(p_home)})")
    print(f"Away win%: {p_away:.3f}  (ML {moneyline_from_prob(p_away)})")
    print(f"Total λ ≈ {avg['mu_total']:.3f}")
