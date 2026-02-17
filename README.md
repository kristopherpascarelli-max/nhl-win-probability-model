# NHL Win Probability Model (Demo Version)

A modular NHL game simulation engine built in Python that converts team-level expected goal rates into probabilistic win and scoring projections.

This public repository contains a sanitized demo version of the forecasting engine. Optional production modules (rest adjustments, goalie modifiers, pace factors, injury adjustments) are not required for the demo and default to neutral values if absent.

---

## Overview

This engine transforms team expected goal rates (xGF/60, xGA/60) into:

- Win probability projections
- Fair moneyline prices
- Expected goal totals
- Simulated game-level scoring distributions

The model uses:

- 5v5 expected goal rates
- Special teams (5v4 / 4v5) adjustments
- Monte Carlo simulation

Optional modules can be layered on top for production use.

---

## Required Data

The demo version requires only:

data/NHL_5v5_2025_26.csv  
data/NHL_5v4_2025_26.csv  
data/NHL_4v5_2025_26.csv  

If optional files such as:

- rest_factors.csv
- goalie_multipliers.csv
- pace_factors.csv
- home_ice.csv
- injuries.csv

are not present, the engine uses neutral defaults and continues execution.

---

## Usage

From the repo root:

python NHL_Simulation.py BOS MTL

Optional arguments:

--sims 50000  
--pp-minutes 10  
--no-ot  
--seed 42  

---

## Example Output

Matchup: MTL @ BOS

Win%  BOS: 58.74%   (fair ML -142)  
Win%  MTL: 41.26%   (fair ML +142)

Expected goals:  
BOS 3.87 | MTL 3.31  
Total: 7.17  

---

## Architecture

The public demo is structured as:

- Core engine (required xG inputs)
- Optional modules layered safely with neutral fallbacks
- Monte Carlo simulation layer
- CLI interface

The production version includes expanded player-level modeling and dynamic parameter calibration.

---

## Tech Stack

- Python 3
- NumPy
- Pandas
- Monte Carlo simulation framework

---

## License

MIT


