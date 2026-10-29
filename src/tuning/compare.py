"""Grid search vs random search (Task 1.2): the comparison table, built from the two results.csv files.

The spec compares the methods on best val score, number of trials and total wall-clock time.
We add hours_to_best (the time spent until the search found its best trial), because it shows
how quickly a method reaches a good setting, not only where it ends up.

Rules that keep the comparison fair:
    - Only successful trials count. A failed attempt is caused by the infrastructure (e.g. a W&B
      upload crash), not by the trial's settings, so it is listed separately (failed_attempts,
      failed_min) instead of being added to the method's time.
    - Time is each trial's wall_time_s (model build + training + W&B), added up in the order the
      trials ran. So hours_to_best depends on the trial order: the grid runs its combinations in
      a fixed order (last key changes fastest), while random search's order is random.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.tuning.run import results_file
from src.utils.config import TuneConfig


@dataclass
class SearchResults:
    """One finished search: its method, the settings it searched, and its results.csv rows."""

    method: str  # "grid" or "random"
    space: dict  # `space` of the tuning config: searched setting -> list of values (grid) or range (random)
    trials: pd.DataFrame  # results.csv rows in the order they ran, failed attempts included

    @property
    def ok(self) -> pd.DataFrame:
        """Successful trials only, in run order."""
        return self.trials[self.trials["status"] == "ok"].reset_index(drop=True)


def load_search(cfg: TuneConfig, results_dir: str | Path | None = None) -> SearchResults:
    """Read the results.csv that `python -m src.cli tune` wrote for this tuning config."""
    path = results_file(cfg, results_dir)
    if not path.exists():
        raise FileNotFoundError(f"No results for {cfg.experiment}: {path} does not exist (run the search first)")
    return SearchResults(cfg.method, cfg.space, pd.read_csv(path))


def best_so_far(search: SearchResults, metric: str, mode: str = "max") -> pd.DataFrame:
    """Progress of a search: after each successful trial, the time spent so far and the best score so far.

    Returns one row per successful trial, in run order, with columns:
        run_name, hours (cumulative, successful trials only), value (this trial's score),
        best (best score up to and including this trial).
    """
    ok = search.ok
    if ok.empty:
        raise ValueError(f"The {search.method} search has no successful trials")
    curve = pd.DataFrame({
        "run_name": ok["run_name"],
        "hours": ok["wall_time_s"].cumsum() / 3600,
        "value": ok[metric],
    })
    curve["best"] = curve["value"].cummax() if mode == "max" else curve["value"].cummin()
    return curve


def comparison_table(searches: list[SearchResults], metric: str, mode: str = "max") -> pd.DataFrame:
    """One row per search method.

    Columns: method, settings_searched, trials (successful), failed_attempts, <metric> (best score),
    best_trial, hours_to_best, total_hours (successful trials), min_per_trial (mean), failed_min.
    """
    rows = []
    for s in searches:
        curve = best_so_far(s, metric, mode)
        i = curve["value"].idxmax() if mode == "max" else curve["value"].idxmin()  # first trial with the best score
        failed = s.trials[s.trials["status"] != "ok"]
        rows.append({
            "method": s.method,
            "settings_searched": len(s.space),
            "trials": len(curve),
            "failed_attempts": len(failed),
            metric: curve.loc[i, "value"],
            "best_trial": curve.loc[i, "run_name"],
            "hours_to_best": curve.loc[i, "hours"],
            "total_hours": curve["hours"].iloc[-1],
            "min_per_trial": s.ok["wall_time_s"].mean() / 60,
            "failed_min": failed["wall_time_s"].sum() / 60,
        })
    return pd.DataFrame(rows)
