"""Load, correct and validate the Man Utd 2025-26 match data.

Reads the "MU Match Data" and "Opponent Data" sheets, assigns the manager from
the match date, applies documented corrections, validates with pandera and
caches the result to parquet. The source workbook is never modified.

Blank cells stay NaN. A zero in the workbook is a real zero.
This module must not import streamlit; the app wraps it in st.cache_data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "man_utd_2025_26_full_stats.xlsx"
CACHE_DIR = ROOT / ".cache"
CACHE_VERSION = "1"  # bump when loading logic changes

MU_SHEET = "MU Match Data"
OPP_SHEET = "Opponent Data"
HEADER_ROW = 3  # 0-indexed, so Excel row 4
N_GAMES = 38

AMORIM = "Ruben Amorim"
FLETCHER = "Darren Fletcher"
CARRICK = "Michael Carrick"
MANAGERS = (AMORIM, FLETCHER, CARRICK)
AMORIM_LAST_DATE = pd.Timestamp("2026-01-04")
CARRICK_FIRST_DATE = pd.Timestamp("2026-01-17")

# ---------------------------------------------------------------------------
# Column layout (identical in both sheets)
# ---------------------------------------------------------------------------
MATCH_COLS = (
    "GW", "Date", "Manager", "Opponent", "H/A", "Stadium",
    "MU goals", "Opp goals", "Result", "Points",
)
PREMATCH_COLS = (
    "MU position", "MU points", "MU form (last 5)", "MU avg rating",
    "Opp position", "Opp points", "Opp form (last 5)", "Opp avg rating",
)
TRACKING_COLS = ("Distance covered (km)", "Sprints")
BIG_CHANCE_COLS = ("Big chances", "Big chances scored", "Big chances missed")
FLOAT_STAT_COLS = ("xG", "xG on target (xGOT)", "Goals prevented")
PCT_COLS = (
    "Possession %", "Duels won %", "Ground duels %", "Aerial duels %",
    "Dribbles %", "Passes in final third %", "Long balls %", "Crosses %",
    "Tackles won %", "Pass accuracy %",
)
COUNT_COLS = (
    "Big chances", "Total shots", "GK saves", "Corners", "Fouls", "Passes",
    "Tackles", "Free kicks", "Yellow cards", "Red cards", "Shots on target",
    "Hit woodwork", "Shots off target", "Blocked shots", "Shots inside box",
    "Shots outside box", "Big chances scored", "Big chances missed",
    "Through balls", "Touches in opp box", "Fouled in final third", "Offsides",
    "Dispossessed", "Ground duels won", "Ground duels total",
    "Aerial duels won", "Aerial duels total", "Dribbles successful",
    "Dribbles attempted", "Accurate passes", "Throw-ins",
    "Final third entries", "Passes in final third accurate",
    "Passes in final third total", "Long balls accurate", "Long balls total",
    "Crosses accurate", "Crosses total", "Total tackles", "Interceptions",
    "Recoveries", "Clearances", "Errors leading to shot",
    "Errors leading to goal", "Big saves", "High claims", "Punches",
    "Goal kicks", "Penalty saves",
)
STAT_COLS = TRACKING_COLS + FLOAT_STAT_COLS + PCT_COLS + COUNT_COLS
ALL_COLS = MATCH_COLS + PREMATCH_COLS + STAT_COLS

# Blanks that are known and accepted: {column: GWs allowed to be NaN}.
# GW1 has no pre-match table; GW33 has no Sofascore tracking data.
# Any other blank fails validation so it can be confirmed before use.
EXPECTED_BLANKS: dict[str, frozenset[int]] = {
    **{c: frozenset({1}) for c in PREMATCH_COLS},
    **{c: frozenset({33}) for c in TRACKING_COLS},
}


@dataclass(frozen=True)
class Correction:
    """A documented override of one cell, applied in code, never in the xlsx."""

    sheet: str
    gw: int
    column: str
    old: float
    new: float
    reason: str


CORRECTIONS: tuple[Correction, ...] = (
    Correction(
        sheet=MU_SHEET,
        gw=26,
        column="Big chances",
        old=1,
        new=2,
        reason=(
            "Sofascore lists 1 big chance but 1 scored and 1 missed. One big "
            "chance was ruled offside; confirmed by the user as 2 created."
        ),
    ),
)


@dataclass(frozen=True)
class MatchData:
    """Man Utd values and opponent values, one row per GW, aligned on GW."""

    mu: pd.DataFrame
    opp: pd.DataFrame


# ---------------------------------------------------------------------------
# Manager assignment
# ---------------------------------------------------------------------------
def assign_manager(match_date: pd.Timestamp) -> str:
    """Return the manager in charge on a given match date."""
    ts = pd.Timestamp(match_date)
    if ts <= AMORIM_LAST_DATE:
        return AMORIM
    if ts < CARRICK_FIRST_DATE:
        return FLETCHER
    return CARRICK


def add_manager(df: pd.DataFrame) -> pd.DataFrame:
    """Set Manager from Date, failing if it disagrees with the sheet's label."""
    out = df.copy()
    derived = out["Date"].map(assign_manager)
    if "Manager" in out.columns:
        mismatch = out["Manager"].notna() & (out["Manager"] != derived)
        if mismatch.any():
            gws = out.loc[mismatch, "GW"].tolist()
            raise ValueError(f"Sheet Manager disagrees with date rule at GW {gws}")
    out["Manager"] = derived
    return out


# ---------------------------------------------------------------------------
# Reading and corrections
# ---------------------------------------------------------------------------
def check_columns(df: pd.DataFrame, sheet: str) -> None:
    """Fail loudly, naming the columns, if any expected column is missing."""
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{sheet!r} is missing columns: {missing}")
    missing_bc = [c for c in BIG_CHANCE_COLS if c not in df.columns]
    if missing_bc:
        raise KeyError(f"{sheet!r} is missing big chance columns: {missing_bc}")


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Read one match sheet as published, with blanks kept as NaN."""
    df = pd.read_excel(path, sheet_name=sheet, header=HEADER_ROW)
    df = df.dropna(how="all")
    check_columns(df, sheet)
    df = df.loc[:, list(ALL_COLS)]
    for col in ("Manager", "Opponent", "H/A", "Stadium", "Result"):
        df[col] = df[col].str.strip()
    return df.sort_values("GW").reset_index(drop=True)


def apply_corrections(
    df: pd.DataFrame, sheet: str, corrections: tuple[Correction, ...] = CORRECTIONS
) -> pd.DataFrame:
    """Apply overrides for this sheet, failing if the source value has changed."""
    out = df.copy()
    for c in corrections:
        if c.sheet != sheet:
            continue
        rows = out.index[out["GW"] == c.gw]
        if len(rows) != 1:
            raise ValueError(f"Correction target GW{c.gw} not found once in {sheet!r}")
        current = out.at[rows[0], c.column]
        if current != c.old:
            raise ValueError(
                f"Correction for {sheet!r} GW{c.gw} {c.column!r} expected "
                f"{c.old}, found {current}. Review the correction."
            )
        out.at[rows[0], c.column] = c.new
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _blanks_as_expected(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise mask: True where every NaN is in EXPECTED_BLANKS."""
    ok = pd.Series(True, index=df.index)
    for col in df.columns:
        allowed = EXPECTED_BLANKS.get(col, frozenset())
        ok &= df[col].notna() | df["GW"].isin(allowed)
    return ok


def result_from_score(gf: pd.Series, ga: pd.Series) -> pd.Series:
    """W/D/L from Man Utd's perspective."""
    return pd.Series(
        np.select([gf > ga, gf == ga], ["W", "D"], "L"), index=gf.index
    )


def build_schema(goals_col: str) -> pa.DataFrameSchema:
    """Pandera schema for one match sheet. goals_col is that team's goals."""
    nonneg = pa.Check.ge(0)
    columns: dict[str, pa.Column] = {
        "GW": pa.Column(int, pa.Check.in_range(1, N_GAMES), unique=True),
        "Date": pa.Column("datetime64[us]", coerce=True),
        "Manager": pa.Column(str, pa.Check.isin(MANAGERS)),
        "Opponent": pa.Column(str),
        "H/A": pa.Column(str, pa.Check.isin(["H", "A"])),
        "Stadium": pa.Column(str),
        "MU goals": pa.Column(int, nonneg),
        "Opp goals": pa.Column(int, nonneg),
        "Result": pa.Column(str, pa.Check.isin(["W", "D", "L"])),
        "Points": pa.Column(int, pa.Check.isin([0, 1, 3])),
        "MU position": pa.Column(float, pa.Check.in_range(1, 20), nullable=True),
        "Opp position": pa.Column(float, pa.Check.in_range(1, 20), nullable=True),
        "MU points": pa.Column(float, nonneg, nullable=True),
        "Opp points": pa.Column(float, nonneg, nullable=True),
        "MU form (last 5)": pa.Column(
            str, pa.Check.str_matches(r"^[WDL]{1,5}$"), nullable=True
        ),
        "Opp form (last 5)": pa.Column(
            str, pa.Check.str_matches(r"^[WDL]{1,5}$"), nullable=True
        ),
        "MU avg rating": pa.Column(float, pa.Check.in_range(0, 10), nullable=True),
        "Opp avg rating": pa.Column(float, pa.Check.in_range(0, 10), nullable=True),
        "Goals prevented": pa.Column(float),
    }
    columns |= {c: pa.Column(float, nonneg, nullable=True) for c in TRACKING_COLS}
    columns |= {c: pa.Column(float, nonneg) for c in ("xG", "xG on target (xGOT)")}
    columns |= {c: pa.Column(float, pa.Check.in_range(0, 1)) for c in PCT_COLS}
    columns |= {c: pa.Column(int, nonneg) for c in COUNT_COLS}

    return pa.DataFrameSchema(
        columns,
        strict=True,
        checks=[
            pa.Check(lambda d: len(d) == N_GAMES, error=f"expected {N_GAMES} rows"),
            pa.Check(
                lambda d: d["GW"].tolist() == list(range(1, N_GAMES + 1)),
                error="GW must run 1..38 in order",
            ),
            pa.Check(
                lambda d: d["Result"] == result_from_score(d["MU goals"], d["Opp goals"]),
                error="Result must match the scoreline",
            ),
            pa.Check(
                lambda d: d["Points"] == d["Result"].map({"W": 3, "D": 1, "L": 0}),
                error="Points must match Result",
            ),
            pa.Check(
                lambda d: (d["H/A"] == "H") == (d["Stadium"] == "Old Trafford"),
                error="H/A must be H exactly when Stadium is Old Trafford",
            ),
            pa.Check(
                lambda d: d["Big chances scored"] + d["Big chances missed"]
                == d["Big chances"],
                error="Big chances scored + missed must equal Big chances",
            ),
            pa.Check(
                lambda d: d["Big chances scored"] <= d[goals_col],
                error=f"Big chances scored cannot exceed {goals_col}",
            ),
            pa.Check(_blanks_as_expected, error="unexpected blank cell"),
        ],
    )


def validate(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Validate one sheet. Raises pandera SchemaErrors listing every failure."""
    goals_col = "MU goals" if sheet == MU_SHEET else "Opp goals"
    return build_schema(goals_col).validate(df, lazy=True)


def check_alignment(mu: pd.DataFrame, opp: pd.DataFrame) -> None:
    """Both sheets must describe the same matches with the same context."""
    shared = list(MATCH_COLS + PREMATCH_COLS)
    if not mu[shared].equals(opp[shared]):
        raise ValueError("MU and Opponent sheets disagree on match/pre-match columns")
    if ((mu["Possession %"] + opp["Possession %"] - 1).abs() > 0.011).any():
        raise ValueError("MU and opponent possession do not sum to 100%")


# ---------------------------------------------------------------------------
# Public entry point with parquet cache
# ---------------------------------------------------------------------------
def cache_key(path: Path) -> str:
    """Hash of workbook bytes, corrections and loader version."""
    h = hashlib.sha256(path.read_bytes())
    h.update(repr(CORRECTIONS).encode())
    h.update(CACHE_VERSION.encode())
    return h.hexdigest()[:12]


def _build(path: Path) -> MatchData:
    frames = {}
    for sheet in (MU_SHEET, OPP_SHEET):
        df = read_sheet(path, sheet)
        df = add_manager(df)
        df = apply_corrections(df, sheet)
        frames[sheet] = validate(df, sheet)
    check_alignment(frames[MU_SHEET], frames[OPP_SHEET])
    return MatchData(mu=frames[MU_SHEET], opp=frames[OPP_SHEET])


def load_match_data(
    path: Path = DATA_PATH, cache_dir: Path | None = CACHE_DIR
) -> MatchData:
    """Load both match sheets, validated, using a parquet cache when available.

    Pass cache_dir=None to always read the workbook.
    """
    if cache_dir is None:
        return _build(path)
    key = cache_key(path)
    mu_file = cache_dir / f"mu_{key}.parquet"
    opp_file = cache_dir / f"opp_{key}.parquet"
    if mu_file.exists() and opp_file.exists():
        mu = validate(pd.read_parquet(mu_file), MU_SHEET)
        opp = validate(pd.read_parquet(opp_file), OPP_SHEET)
        return MatchData(mu=mu, opp=opp)
    data = _build(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.mu.to_parquet(mu_file, index=False)
    data.opp.to_parquet(opp_file, index=False)
    return data
