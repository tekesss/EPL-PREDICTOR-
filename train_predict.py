#!/usr/bin/env python3
"""
EPL match model: Dixon-Coles ratings + LightGBM (Poisson) -> score grid.

Pipeline
  1. build_matches()     join Understat fixtures (xG, upcoming games) with
                         football-data.co.uk (shots, Pinnacle closing odds).
  2. rolling_dc()        Dixon-Coles style attack/defence ratings, refitted
                         before every match date using ONLY earlier matches
                         (time-decayed Poisson GLM). Two variants: on goals
                         and on xG.
  3. team_features()     pre-match exponentially weighted form per team.
  4. LightGBM            one Poisson model for goals, one for shots, trained
                         on a "long" table (one row per team per match).
  5. score_grid()        Poisson x Poisson with the Dixon-Coles low-score
                         correction (rho) -> 1X2, exact scores, over/under.

Usage
  python train_predict.py               # train on all played matches, predict upcoming
  python train_predict.py --backtest    # also run walk-forward evaluation per season

Outputs (in ./predictions/)
  predictions.json   upcoming fixtures with probabilities, xG, shots, top scores
  backtest.json      per-season log-loss / RPS vs baselines (only with --backtest)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=UserWarning)

RAW = Path("data/raw")
OUT = Path("predictions")

# football-data.co.uk name -> Understat name
NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
}

MAX_GOALS = 10                 # score grid is (MAX_GOALS+1) x (MAX_GOALS+1)
DC_WINDOW_DAYS = 3 * 365       # Dixon-Coles fit window
DC_HALF_LIFE_DAYS = 365        # time decay: weight = 0.5 ** (age / half_life)
EWM_HALF_LIFE = 8              # form features: half-life in matches
FIRST_TEST_SEASON = "2018-19"  # earlier seasons are warm-up for features

LGB_PARAMS = dict(
    objective="poisson",
    learning_rate=0.03,
    n_estimators=350,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


# --------------------------------------------------------------------------- #
# 1. data
# --------------------------------------------------------------------------- #
def build_matches() -> pd.DataFrame:
    """One row per fixture (played and upcoming)."""
    fx = pd.read_parquet(RAW / "understat_fixtures.parquet")
    num = ["home_goals", "away_goals", "home_xg", "away_xg"]
    fx[num] = fx[num].apply(pd.to_numeric, errors="coerce")
    fx = fx.drop(columns=["p_home", "p_draw", "p_away"])   # post-match, not a forecast
    fx["date"] = fx["datetime"].dt.normalize()

    fd = pd.read_parquet(RAW / "football_data_e0.parquet")
    fd["home_team"] = fd["HomeTeam"].replace(NAME_MAP)
    fd["away_team"] = fd["AwayTeam"].replace(NAME_MAP)
    keep = {"HS": "home_shots", "AS": "away_shots",
            "HST": "home_sot", "AST": "away_sot",
            "PSCH": "ps_close_h", "PSCD": "ps_close_d", "PSCA": "ps_close_a"}
    fd = fd[["season", "home_team", "away_team", *keep]].rename(columns=keep)

    # A pair meets at a given home ground exactly once per season -> unique key.
    m = fx.merge(fd, on=["season", "home_team", "away_team"], how="left",
                 validate="one_to_one")

    played = m[m["is_result"]]
    unmatched = played["home_shots"].isna()
    # football-data lags a few days behind Understat; anything older is a bug.
    stale = played.loc[unmatched, "date"] < played["date"].max() - pd.Timedelta(days=35)
    if stale.any():
        bad = played.loc[unmatched][stale][["season", "home_team", "away_team"]]
        raise SystemExit(f"Join failed for {len(bad)} matches, e.g.\n{bad.head()}")

    # Per-team advanced stats (PPDA, deep completions) from Understat team history.
    us = pd.read_parquet(RAW / "understat_team_matches.parquet")
    us["date"] = pd.to_datetime(us["date"]).dt.normalize()
    us["ppda"] = us["ppda"].map(lambda d: d["att"] / d["def"] if d and d["def"] else np.nan)
    us["ppda_allowed"] = us["ppda_allowed"].map(
        lambda d: d["att"] / d["def"] if d and d["def"] else np.nan)
    us = us[["team", "date", "ppda", "ppda_allowed", "deep", "deep_allowed"]]
    for side in ("home", "away"):
        m = m.merge(us.rename(columns={"team": f"{side}_team",
                                       **{c: f"{side}_{c}" for c in us.columns[2:]}}),
                    on=[f"{side}_team", "date"], how="left")

    return m.sort_values(["datetime", "match_id"]).reset_index(drop=True)


def to_long(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one from each team's point of view."""
    cols = ["goals", "xg", "shots", "sot", "ppda", "ppda_allowed", "deep", "deep_allowed"]
    rows = []
    for side, opp, is_home in (("home", "away", 1), ("away", "home", 0)):
        d = pd.DataFrame({
            "match_id": m["match_id"], "season": m["season"], "date": m["date"],
            "is_result": m["is_result"], "is_home": is_home,
            "team": m[f"{side}_team"], "opp": m[f"{opp}_team"],
        })
        for c in ("goals", "xg", "shots", "sot"):
            d[f"{c}_for"] = m[f"{side}_{c}"]
            d[f"{c}_against"] = m[f"{opp}_{c}"]
        for c in ("ppda", "ppda_allowed", "deep", "deep_allowed"):
            d[c] = m[f"{side}_{c}"]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. rolling Dixon-Coles ratings
# --------------------------------------------------------------------------- #
DC_RIDGE = 1e-3   # L2 penalty: shrinks ratings of teams with few matches toward average


class DCFit:
    """
    Weighted Poisson GLM  log(lambda) = b0 + home*is_home + att[team] - def[opp]
    solved by Newton's method. The model has only ~2n+2 parameters, so each
    Newton step is a tiny linear solve; warm-starting from the previous
    match date's solution means 2-4 steps instead of hundreds.
    """

    def __init__(self):
        self.prev: dict[str, float] = {}   # parameter name -> value (warm start)

    def fit(self, hist: pd.DataFrame, target: str, ref_date: pd.Timestamp):
        teams = sorted(set(hist["team"]) | set(hist["opp"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        names = ["b0", "home"] + [f"att:{t}" for t in teams] + [f"def:{t}" for t in teams]

        N = len(hist)
        X = np.zeros((N, 2 * n + 2))
        r = np.arange(N)
        X[:, 0] = 1.0
        X[:, 1] = hist["is_home"].to_numpy()
        X[r, 2 + hist["team"].map(idx).to_numpy()] = 1.0
        X[r, 2 + n + hist["opp"].map(idx).to_numpy()] = -1.0
        y = hist[target].to_numpy(dtype=float)
        age = (ref_date - hist["date"]).dt.days.to_numpy()
        w = 0.5 ** (age / DC_HALF_LIFE_DAYS)
        w = w / w.sum()

        beta = np.array([self.prev.get(k, 0.0) for k in names])
        if "b0" not in self.prev:
            beta[0] = np.log(np.average(y, weights=w))
        pen = np.full(len(beta), DC_RIDGE)
        pen[:2] = 0.0                                  # do not shrink intercept/home

        for _ in range(25):
            mu = np.exp(X @ beta)
            grad = X.T @ (w * (mu - y)) + pen * beta
            H = (X * (w * mu)[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(len(beta))
            step = np.linalg.solve(H, grad)
            beta -= step
            if np.abs(step).max() < 1e-7:
                break

        self.prev = dict(zip(names, beta))
        return self

    def log_lambda(self, team: pd.Series, opp: pd.Series, is_home: pd.Series) -> np.ndarray:
        # Unseen teams (promoted after a long absence) get 0 = league average;
        # the "promoted" feature lets LightGBM correct for that.
        p = self.prev
        return (p["b0"] + p["home"] * is_home.to_numpy()
                + team.map(lambda t: p.get(f"att:{t}", 0.0)).to_numpy()
                - opp.map(lambda t: p.get(f"def:{t}", 0.0)).to_numpy())


def rolling_dc(long: pd.DataFrame) -> pd.DataFrame:
    """
    For every match date D, fit on matches strictly before D and predict
    log-lambda for every team-row on D. Returns columns dc_goals, dc_xg.
    Upcoming fixtures all use the latest fit.
    """
    played = long[long["is_result"]]
    out = pd.Series(np.nan, index=long.index, name="dc_goals"), \
          pd.Series(np.nan, index=long.index, name="dc_xg")

    last_played = played["date"].max()
    dates = sorted(long.loc[long["date"] <= last_played, "date"].unique())
    groups = list(dates) + ["upcoming"]

    fitters = {"goals_for": DCFit(), "xg_for": DCFit()}
    for d in groups:
        if d == "upcoming":
            ref = last_played + pd.Timedelta(days=1)
            rows = long.index[~long["is_result"]]
        else:
            ref = d
            rows = long.index[long["date"] == d]
        hist = played[(played["date"] < ref) & (played["date"] >= ref - pd.Timedelta(days=DC_WINDOW_DAYS))]
        if len(hist) < 400:          # not enough history yet (first season)
            continue
        tr = long.loc[rows]
        for k, target in enumerate(fitters):
            f = fitters[target].fit(hist.dropna(subset=[target]), target, ref)
            out[k].loc[rows] = f.log_lambda(tr["team"], tr["opp"], tr["is_home"])
    return pd.concat(out, axis=1)


# --------------------------------------------------------------------------- #
# 3. form features (pre-match only)
# --------------------------------------------------------------------------- #
FORM_COLS = ["goals_for", "goals_against", "xg_for", "xg_against",
             "shots_for", "shots_against", "sot_for", "sot_against",
             "ppda", "ppda_allowed", "deep", "deep_allowed"]


def team_features(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["team", "date", "match_id"]).copy()
    g = long.groupby("team", group_keys=False)

    # shift(1): a match's own stats never enter its own features (no leakage).
    for c in FORM_COLS:
        long[f"f_{c}"] = g[c].apply(
            lambda s: s.shift(1).ewm(halflife=EWM_HALF_LIFE, ignore_na=True).mean())

    # Upcoming fixtures: carry forward the team's latest form.
    upcoming = ~long["is_result"]
    for c in FORM_COLS:
        last = long[~upcoming].groupby("team")[c].apply(
            lambda s: s.ewm(halflife=EWM_HALF_LIFE, ignore_na=True).mean().iloc[-1])
        long.loc[upcoming, f"f_{c}"] = long.loc[upcoming, "team"].map(last)

    # Rest days come from the fixture list, which is known in advance,
    # so they are valid for upcoming games too.
    prev_date = g["date"].shift(1)
    long["rest_days"] = (long["date"] - prev_date).dt.days.clip(upper=21)
    # Matches already PLAYED before this one (upcoming games do not count).
    long["n_prior"] = (g["is_result"].cumsum() - long["is_result"].astype(int)).clip(upper=60)

    seasons = sorted(long["season"].unique())
    in_season = long.groupby("season")["team"].apply(set).to_dict()
    long["promoted"] = [
        int(i > 0 and t not in in_season[seasons[i - 1]])
        for t, i in zip(long["team"], long["season"].map(seasons.index))
    ]
    return long.sort_index()


def add_opponent(long: pd.DataFrame) -> pd.DataFrame:
    own = ["dc_goals", "dc_xg", "rest_days", "promoted", "n_prior"] + [f"f_{c}" for c in FORM_COLS]
    opp = long[["match_id", "team", *own]].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in own}})
    return long.merge(opp, on=["match_id", "opp"], how="left")


def feature_names(long: pd.DataFrame) -> list[str]:
    base = ["is_home", "dc_goals", "dc_xg", "rest_days", "promoted", "n_prior"] + \
           [f"f_{c}" for c in FORM_COLS]
    return base + [f"opp_{c}" for c in base if c != "is_home"]


# --------------------------------------------------------------------------- #
# 4/5. models and score grid
# --------------------------------------------------------------------------- #
def fit_lgb(train: pd.DataFrame, feats: list[str], target: str) -> lgb.LGBMRegressor:
    t = train.dropna(subset=[target])
    model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=42)
    model.fit(t[feats], t[target])
    return model


def score_grid(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """(n, G, G) array of P(home=h, away=a) with Dixon-Coles low-score correction."""
    k = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(k[None, :], lh[:, None])
    pa = poisson.pmf(k[None, :], la[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    grid[:, 0, 0] *= 1 - lh * la * rho
    grid[:, 0, 1] *= 1 + lh * rho
    grid[:, 1, 0] *= 1 + la * rho
    grid[:, 1, 1] *= 1 - rho
    grid = np.clip(grid, 0, None)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def outcome_probs(grid: np.ndarray) -> np.ndarray:
    home = np.tril(np.ones(grid.shape[1:]), -1)   # h > a
    draw = np.eye(grid.shape[1])
    return np.stack([(grid * home).sum((1, 2)), (grid * draw).sum((1, 2)),
                     (grid * home.T).sum((1, 2))], axis=1)


def outcome_masks(G: int) -> list[np.ndarray]:
    home = np.tril(np.ones((G, G)), -1)        # h > a
    return [home, np.eye(G), home.T]


def pick_scores(grid: np.ndarray) -> np.ndarray:
    """
    "Pick" = most likely exact score WITHIN the most likely 1X2 outcome.

    The unconditional most likely score is 1-1 whenever both expected-goal
    values sit between 1 and 2 (about two matches in three), so it carries
    little information. The pick is consistent with the 1X2 call; it is
    slightly LESS often exactly right (see backtest), so both are published.
    Returns flat indices into the (G, G) grid.
    """
    G = grid.shape[1]
    masks = outcome_masks(G)
    P = outcome_probs(grid)
    best = P.argmax(1)
    return np.array([(grid[i] * masks[best[i]]).argmax() for i in range(len(grid))])


def fit_rho(lh, la, hg, ag) -> float:
    hg, ag = hg.astype(int), ag.astype(int)
    ok = (hg <= MAX_GOALS) & (ag <= MAX_GOALS)
    lh, la, hg, ag = lh[ok], la[ok], hg[ok], ag[ok]

    def nll(rho):
        g = score_grid(lh, la, rho)
        return -np.log(g[np.arange(len(hg)), hg, ag] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)


def wide_lambdas(long: pd.DataFrame, col: str) -> pd.DataFrame:
    """long rows -> one row per match with home/away value of `col`."""
    h = long[long["is_home"] == 1].set_index("match_id")[col]
    a = long[long["is_home"] == 0].set_index("match_id")[col]
    return pd.DataFrame({"home": h, "away": a})


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(P: np.ndarray, y: np.ndarray) -> dict | None:
    if len(y) == 0:
        return None
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    O = np.eye(3)[y]
    cp, co = np.cumsum(P, 1), np.cumsum(O, 1)
    return {
        "log_loss": float(-np.log(P[np.arange(len(y)), y]).mean()),
        "rps": float((((cp - co)[:, :2]) ** 2).sum(1).mean() / 2),
        "accuracy": float((P.argmax(1) == y).mean()),
        "n": int(len(y)),
    }


def outcome_index(hg, ag) -> np.ndarray:
    return np.select([hg > ag, hg == ag], [0, 1], 2)


# --------------------------------------------------------------------------- #
def prepare() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    m = build_matches()
    long = to_long(m)
    print(f"  {m['is_result'].sum():,} played, {(~m['is_result']).sum():,} upcoming")
    print("  fitting rolling Dixon-Coles ratings ...")
    long = pd.concat([long, rolling_dc(long)], axis=1)
    long = add_opponent(team_features(long))
    return m, long, feature_names(long)


def backtest(m: pd.DataFrame, long: pd.DataFrame, feats: list[str]) -> dict:
    played = long[long["is_result"]]
    seasons = sorted(s for s in played["season"].unique() if s >= FIRST_TEST_SEASON)
    mp = m[m["is_result"]].set_index("match_id")
    report, pooled = {}, {k: [] for k in ("base", "dc", "lgb", "pinnacle", "y", "score_hit", "pick_hit")}

    for s in seasons:
        train = played[(played["season"] < s) & played["dc_goals"].notna()]
        test = played[played["season"] == s]
        mt = mp.loc[mp["season"] == s]
        y = outcome_index(mt["home_goals"].to_numpy(), mt["away_goals"].to_numpy())

        # baseline: historical outcome frequencies
        tr = mp[mp["season"] < s]
        base = np.bincount(outcome_index(tr["home_goals"], tr["away_goals"]), minlength=3) / len(tr)
        P_base = np.tile(base, (len(y), 1))

        # Dixon-Coles alone
        dc_tr = wide_lambdas(train, "dc_goals").pipe(np.exp)
        rho = fit_rho(dc_tr["home"].to_numpy(), dc_tr["away"].to_numpy(),
                      mp.loc[dc_tr.index, "home_goals"].to_numpy(),
                      mp.loc[dc_tr.index, "away_goals"].to_numpy())
        dc_te = wide_lambdas(test, "dc_goals").pipe(np.exp).loc[mt.index]
        P_dc = outcome_probs(score_grid(dc_te["home"].to_numpy(), dc_te["away"].to_numpy(), rho))

        # LightGBM
        model = fit_lgb(train, feats, "goals_for")
        te = test.copy()
        te["lam"] = model.predict(te[feats])
        lam = wide_lambdas(te, "lam").loc[mt.index]
        grid = score_grid(lam["home"].to_numpy(), lam["away"].to_numpy(), rho)
        P_lgb = outcome_probs(grid)
        top = grid.reshape(len(grid), -1).argmax(1)
        hg, ag = mt["home_goals"].to_numpy(), mt["away_goals"].to_numpy()
        hit = (top // (MAX_GOALS + 1) == hg) & (top % (MAX_GOALS + 1) == ag)
        pk = pick_scores(grid)
        pick_hit = (pk // (MAX_GOALS + 1) == hg) & (pk % (MAX_GOALS + 1) == ag)

        # Pinnacle closing odds (proportional de-vig)
        odds = mt[["ps_close_h", "ps_close_d", "ps_close_a"]].to_numpy(dtype=float)
        has = ~np.isnan(odds).any(1)
        P_pin = (1 / odds[has]) / (1 / odds[has]).sum(1, keepdims=True)

        report[s] = {
            "base_rates": metrics(P_base, y),
            "dixon_coles": metrics(P_dc, y),
            "lightgbm": metrics(P_lgb, y),
            "lightgbm_on_pinnacle_matches": metrics(P_lgb[has], y[has]),
            "pinnacle_close": metrics(P_pin, y[has]),
            "exact_score_hit_rate": float(hit.mean()),
            "pick_score_hit_rate": float(pick_hit.mean()),
            "rho": rho,
        }
        r = report[s]
        print(f"  {s}  RPS  base {r['base_rates']['rps']:.4f}  DC {r['dixon_coles']['rps']:.4f}"
              f"  LGB {r['lightgbm']['rps']:.4f}  Pinnacle "
              f"{r['pinnacle_close']['rps'] if r['pinnacle_close'] else float('nan'):.4f}"
              f"  | exact score hit {hit.mean():.1%}")

        pooled["base"].append(P_base); pooled["dc"].append(P_dc); pooled["lgb"].append(P_lgb)
        pooled["pinnacle"].append(_fill(P_pin, has))
        pooled["y"].append(y); pooled["score_hit"].append(hit); pooled["pick_hit"].append(pick_hit)

    y = np.concatenate(pooled["y"])
    pin = np.vstack(pooled["pinnacle"]); has = ~np.isnan(pin).any(1)
    lgbP = np.vstack(pooled["lgb"])
    report["ALL"] = {
        "base_rates": metrics(np.vstack(pooled["base"]), y),
        "dixon_coles": metrics(np.vstack(pooled["dc"]), y),
        "lightgbm": metrics(lgbP, y),
        "lightgbm_on_pinnacle_matches": metrics(lgbP[has], y[has]),
        "pinnacle_close": metrics(pin[has], y[has]),
        "exact_score_hit_rate": float(np.concatenate(pooled["score_hit"]).mean()),
        "pick_score_hit_rate": float(np.concatenate(pooled["pick_hit"]).mean()),
    }
    return report


def _fill(P_sub: np.ndarray, mask: np.ndarray) -> np.ndarray:
    full = np.full((len(mask), 3), np.nan)
    full[mask] = P_sub
    return full


def predict_upcoming(m: pd.DataFrame, long: pd.DataFrame, feats: list[str]) -> list[dict]:
    played = long[long["is_result"] & long["dc_goals"].notna()]
    goals_model = fit_lgb(played, feats, "goals_for")
    shots_model = fit_lgb(played, feats, "shots_for")
    sot_model = fit_lgb(played, feats, "sot_for")

    mp = m[m["is_result"]].set_index("match_id")
    dc = wide_lambdas(played, "dc_goals").pipe(np.exp)
    rho = fit_rho(dc["home"].to_numpy(), dc["away"].to_numpy(),
                  mp.loc[dc.index, "home_goals"].to_numpy(),
                  mp.loc[dc.index, "away_goals"].to_numpy())

    up = long[~long["is_result"]].copy()
    up["lam"] = goals_model.predict(up[feats])
    up["shots"] = shots_model.predict(up[feats])
    up["sot"] = sot_model.predict(up[feats])
    lam = wide_lambdas(up, "lam")
    shots = wide_lambdas(up, "shots")
    sot = wide_lambdas(up, "sot")

    fx = m[~m["is_result"]].set_index("match_id").loc[lam.index]
    grid = score_grid(lam["home"].to_numpy(), lam["away"].to_numpy(), rho)
    P = outcome_probs(grid)
    G = MAX_GOALS + 1
    picks = pick_scores(grid)
    masks = outcome_masks(G)
    M = 6                                        # score matrix shown on the site: 0..5 goals

    def sc(k):                                   # flat index -> "h-a"
        return f"{k // G}-{k % G}"

    out = []
    for i, mid in enumerate(lam.index):
        flat = grid[i].ravel()
        order = np.argsort(flat)[::-1][:5]
        total = np.add.outer(np.arange(G), np.arange(G))
        out.append({
            "match_id": str(mid),
            "kickoff": fx.loc[mid, "datetime"].isoformat(),
            "home_team": fx.loc[mid, "home_team"],
            "away_team": fx.loc[mid, "away_team"],
            "p_home": round(float(P[i, 0]), 4),
            "p_draw": round(float(P[i, 1]), 4),
            "p_away": round(float(P[i, 2]), 4),
            "xg_home": round(float(lam.loc[mid, "home"]), 2),
            "xg_away": round(float(lam.loc[mid, "away"]), 2),
            "shots_home": round(float(shots.loc[mid, "home"]), 1),
            "shots_away": round(float(shots.loc[mid, "away"]), 1),
            "sot_home": round(float(sot.loc[mid, "home"]), 1),
            "sot_away": round(float(sot.loc[mid, "away"]), 1),
            "p_over_2_5": round(float(grid[i][total > 2.5].sum()), 4),
            "p_btts": round(float(grid[i][1:, 1:].sum()), 4),
            "pick": {"score": sc(picks[i]), "p": round(float(flat[picks[i]]), 4)},
            "score_by_outcome": {
                name: {"score": sc(int((grid[i] * masks[j]).argmax())),
                       "p": round(float((grid[i] * masks[j]).max()), 4)}
                for j, name in enumerate(("home", "draw", "away"))
            },
            "score_matrix": np.round(grid[i][:M, :M], 4).tolist(),
            "top_scores": [
                {"score": f"{k // G}-{k % G}", "p": round(float(flat[k]), 4)} for k in order
            ],
        })
    return sorted(out, key=lambda r: r["kickoff"])


def version_stamp() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD"),
        "data_commit": git("log", "-1", "--format=%H", "--", "data/raw"),
        "model": "dixon-coles + lightgbm-poisson v1",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true", help="run walk-forward evaluation")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print("[1/3] preparing data and features")
    m, long, feats = prepare()

    if args.backtest:
        print("[2/3] walk-forward backtest")
        rep = backtest(m, long, feats)
        a = rep["ALL"]
        print(f"  ALL  RPS base {a['base_rates']['rps']:.4f}  DC {a['dixon_coles']['rps']:.4f}"
              f"  LGB {a['lightgbm']['rps']:.4f}  | on Pinnacle matches: LGB "
              f"{a['lightgbm_on_pinnacle_matches']['rps']:.4f} vs Pinnacle {a['pinnacle_close']['rps']:.4f}")
        (OUT / "backtest.json").write_text(json.dumps({**version_stamp(), "seasons": rep}, indent=2))

    print("[3/3] training on all played matches and predicting upcoming")
    preds = predict_upcoming(m, long, feats)
    (OUT / "predictions.json").write_text(
        json.dumps({**version_stamp(), "matches": preds}, indent=2, ensure_ascii=False))
    print(f"  wrote {len(preds)} upcoming fixtures to {OUT / 'predictions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
