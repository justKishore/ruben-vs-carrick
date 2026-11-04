"""Tests for src/load.py: manager assignment, blanks, corrections, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors as pe
import pytest

from src.load import (
    AMORIM,
    BIG_CHANCE_COLS,
    CARRICK,
    FLETCHER,
    MU_SHEET,
    OPP_SHEET,
    TRACKING_COLS,
    Correction,
    MatchData,
    apply_corrections,
    assign_manager,
    check_columns,
    load_match_data,
    validate,
)


@pytest.fixture(scope="module")
def data() -> MatchData:
    return load_match_data(cache_dir=None)


# --- manager assignment ------------------------------------------------------
@pytest.mark.parametrize("sheet", ["mu", "opp"])
def test_manager_counts(data: MatchData, sheet: str) -> None:
    counts = getattr(data, sheet)["Manager"].value_counts()
    assert counts[AMORIM] == 20
    assert counts[FLETCHER] == 1
    assert counts[CARRICK] == 17


def test_manager_gw_ranges(data: MatchData) -> None:
    gw = data.mu.groupby("Manager")["GW"].agg(["min", "max"])
    assert tuple(gw.loc[AMORIM]) == (1, 20)
    assert tuple(gw.loc[FLETCHER]) == (21, 21)
    assert tuple(gw.loc[CARRICK]) == (22, 38)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2025-08-18", AMORIM),
        ("2026-01-04", AMORIM),
        ("2026-01-05", FLETCHER),
        ("2026-01-16", FLETCHER),
        ("2026-01-17", CARRICK),
        ("2026-05-25", CARRICK),
    ],
)
def test_assign_manager_boundaries(day: str, expected: str) -> None:
    assert assign_manager(pd.Timestamp(day)) == expected


# --- blanks are NaN, zeros stay zero ----------------------------------------
@pytest.mark.parametrize("sheet", ["mu", "opp"])
def test_gw33_tracking_is_nan(data: MatchData, sheet: str) -> None:
    df = getattr(data, sheet).set_index("GW")
    for col in TRACKING_COLS:
        assert np.isnan(df.at[33, col]), col
        assert df[col].drop(index=33).notna().all(), col


def test_gw10_mu_fouls_is_real_zero(data: MatchData) -> None:
    assert data.mu.set_index("GW").at[10, "Fouls"] == 0


def test_gw1_prematch_is_nan(data: MatchData) -> None:
    row = data.mu.set_index("GW").loc[1]
    assert pd.isna(row["Opp position"])
    assert pd.isna(row["Opp form (last 5)"])


# --- corrections ---------------------------------------------------------------
def test_gw26_big_chances_corrected(data: MatchData) -> None:
    row = data.mu.set_index("GW").loc[26, list(BIG_CHANCE_COLS)]
    assert row.tolist() == [2, 1, 1]


def test_correction_fails_if_source_changed(data: MatchData) -> None:
    stale = Correction(MU_SHEET, 26, "Big chances", old=5, new=6, reason="test")
    with pytest.raises(ValueError, match="expected 5"):
        apply_corrections(data.mu, MU_SHEET, (stale,))


# --- validation fails loudly ----------------------------------------------------
@pytest.mark.parametrize("col", BIG_CHANCE_COLS)
def test_missing_big_chance_column_raises(data: MatchData, col: str) -> None:
    with pytest.raises(KeyError, match=col):
        check_columns(data.opp.drop(columns=col), OPP_SHEET)


def test_unexpected_blank_fails_validation(data: MatchData) -> None:
    broken = data.mu.copy()
    broken["xG"] = broken["xG"].astype(float)
    broken.loc[broken["GW"] == 5, "xG"] = np.nan
    with pytest.raises(pe.SchemaErrors):
        validate(broken, MU_SHEET)


def test_unexpected_tracking_blank_fails_validation(data: MatchData) -> None:
    broken = data.mu.copy()
    broken.loc[broken["GW"] == 12, "Sprints"] = np.nan
    with pytest.raises(pe.SchemaErrors, match="unexpected blank"):
        validate(broken, MU_SHEET)


def test_inconsistent_big_chances_fail_validation(data: MatchData) -> None:
    broken = data.mu.copy()
    broken.loc[broken["GW"] == 26, "Big chances"] = 1
    with pytest.raises(pe.SchemaErrors, match="Big chances scored"):
        validate(broken, MU_SHEET)


# --- parquet cache ---------------------------------------------------------------
def test_parquet_cache_roundtrip(data: MatchData, tmp_path) -> None:
    first = load_match_data(cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 2
    second = load_match_data(cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first.mu, data.mu)
    pd.testing.assert_frame_equal(second.mu, first.mu)
    pd.testing.assert_frame_equal(second.opp, first.opp)
