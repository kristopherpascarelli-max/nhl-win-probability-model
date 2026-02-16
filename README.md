# nhl-win-probability-model
End-to-end NHL game forecasting engine built in Python that transforms team and player metrics into probabilistic win and scoring projections. 
## Overview

This project is a full-game NHL forecasting engine that converts team-level and player-level performance metrics into:

- Win probability projections
- Fair moneyline prices
- Expected goal totals
- Game-level scoring distributions

The model integrates advanced hockey metrics (xGF/xGA), pace adjustments, and goalie performance modifiers to produce probabilistic outputs suitable for analytics, simulation, and pricing workflows.

---

## Features

- Team strength modeling using expected goal rates
- Goalie impact adjustments
- Probabilistic moneyline generation
- Total goals projection
- CSV export of structured outputs
- Modular architecture for expansion (player-level, special teams, Monte Carlo)

---

## Example Output

The engine generates structured CSV output containing:

- Game date
- Home/Away teams
- Starting goalies
- Win probability (%)
- Fair moneyline
- Projected total goals

---

## Tech Stack

- Python 3
- Pandas
- NumPy
- Custom probabilistic modeling pipeline

---

## Future Enhancements

- Monte Carlo game simulation
- Player-level RAPM integration
- API deployment layer
- Web-based visualization dashboard

---

## License

MIT
