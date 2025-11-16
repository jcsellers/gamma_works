from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from xdte.data.loaders import (
    OptionLeg,
    PortfolioRecord,
    compute_representative_trades,
    parse_timestamp,
    read_portfolio_file,
    tag_trading_session,
)


def _backtest_fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "backtest"


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "portfolio_0dte_sample.csv"
    )
    destination = tmp_path / "portfolio_0dte_sample.csv"
    destination.write_bytes(fixture_path.read_bytes())
    return destination


def test_parse_timestamp_handles_multiple_formats() -> None:
    assert parse_timestamp("2024-05-01T09:31:00").hour == 9
    assert parse_timestamp("2024/05/01 08:59").minute == 59
    assert parse_timestamp("05-01-2024 16:30").hour == 16


def test_parse_timestamp_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        parse_timestamp("2024.05.01 10:00")


def test_tag_trading_session_labels_expected_ranges() -> None:
    regular = parse_timestamp("2024-05-01T10:00:00")
    pre = parse_timestamp("2024/05/01 08:00")
    post = parse_timestamp("05-01-2024 16:15")
    overnight = parse_timestamp("05-01-2024 21:15")

    assert tag_trading_session(regular) == "regular"
    assert tag_trading_session(pre) == "pre"
    assert tag_trading_session(post) == "post"
    assert tag_trading_session(overnight) == "overnight"


def test_read_portfolio_file_normalises_rows(sample_csv: Path) -> None:
    records = list(read_portfolio_file(sample_csv))
    assert len(records) == 3

    first = records[0]
    assert isinstance(first, PortfolioRecord)
    assert first.open_timestamp == datetime(2024, 5, 1, 9, 35)
    assert first.close_timestamp == datetime(2024, 5, 1, 15, 58)
    assert first.open_session == "regular"
    assert first.close_session == "regular"
    assert first.strategy == "test_strategy"
    assert first.tenor == "0dte"
    assert first.context["tenor"] == "0dte"
    assert first.context["source"] == sample_csv.name
    assert first.premium == 240.0
    assert first.pnl == 120.0
    assert first.opening_price == 4200.5
    assert first.closing_price == 4190.3
    assert first.margin_requirement == 5000.0
    assert first.max_loss == -50.0
    assert pytest.approx(first.notional) == 480.0
    assert len(first.legs) == 2
    assert first.legs[0] == OptionLeg(
        expiry=datetime(2024, 5, 1).date(),
        strike=4200.0,
        option_type="C",
        position="STO",
        price=3.5,
    )
    assert first.legs[1].option_type == "C"
    assert first.raw["Legs"].startswith("1 May 24 4200 C")

    second = records[1]
    assert second.open_session == "pre"
    assert second.close_session == "regular"
    assert second.strategy == "alternate_strategy"
    assert second.funds_at_close is None
    assert second.legs == (
        OptionLeg(
            expiry=datetime(2024, 5, 1).date(),
            strike=4180.0,
            option_type="P",
            position="BTO",
            price=2.1,
        ),
    )

    third = records[2]
    assert third.open_session == "post"
    assert third.close_session == "regular"
    assert third.strategy == ""
    assert third.legs == ()


def test_read_portfolio_file_preserves_raw_strings(sample_csv: Path) -> None:
    expected_rows = []
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError("sample fixture missing headers")
        reader.fieldnames = [
            (field.strip() if field is not None else "") for field in reader.fieldnames
        ]
        for row in reader:
            cleaned = {
                (key or "").strip(): (value or "").strip()
                for key, value in row.items()
                if key
            }
            expected_rows.append(cleaned)

    records = list(read_portfolio_file(sample_csv))

    for record, expected in zip(records, expected_rows, strict=True):
        assert record.raw == expected


def test_compute_representative_trades_uses_notional(sample_csv: Path) -> None:
    records = list(read_portfolio_file(sample_csv))

    selections = compute_representative_trades(records)
    assert set(selections) == {
        ("test_strategy", "0dte"),
        ("alternate_strategy", "0dte"),
        ("unknown", "0dte"),
    }

    assert selections[("test_strategy", "0dte")].premium == 240.0

    def group_by_open_session(record: PortfolioRecord) -> str:
        return record.open_session

    def score_by_realised_pnl(record: PortfolioRecord) -> float:
        return abs(record.pnl or 0.0)

    session_selections = compute_representative_trades(
        records, key=group_by_open_session, score=score_by_realised_pnl
    )
    assert set(session_selections) == {"regular", "pre", "post"}
    assert session_selections["regular"].strategy == "test_strategy"
    assert session_selections["pre"].strategy == "alternate_strategy"


def test_read_portfolio_file_requires_core_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio_missing.csv"
    csv_path.write_text(
        "Date Opened,Time Opened,Premium\n2024-05-01,09:35:00,10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required column 'Date Closed'"):
        list(read_portfolio_file(csv_path))


def test_read_portfolio_file_rejects_malformed_legs(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio_bad_leg.csv"
    csv_path.write_text(
        "Date Opened,Time Opened,Opening Price,Legs,Premium,Closing Price,Date Closed,Time Closed,Avg. Closing Cost,Reason For Close,P/L,No. of Contracts,Funds at Close,Margin Req.,Strategy,Opening Short/Long Ratio,Closing Short/Long Ratio,Opening VIX,Closing VIX,Gap,Movement,Max Profit,Max Loss\n"
        "2024-05-01,09:35:00,4200.5,bad leg,10,4190.3,2024-05-01,10:00:00,0,Expired,5,1,1000,500,test_strategy,,,,,,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unrecognised leg format"):
        list(read_portfolio_file(csv_path))


def test_read_portfolio_file_rejects_invalid_numeric(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio_invalid_numeric.csv"
    csv_path.write_text(
        "Date Opened,Time Opened,Opening Price,Legs,Premium,Closing Price,Date Closed,Time Closed,P/L,Strategy\n"
        "2024-05-01,09:30:00,4200.5,1 May 24 4200 C STO 3.50,not_a_number,4190.3,2024-05-01,15:58:00,120,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unable to parse string"):
        list(read_portfolio_file(csv_path))


def test_backtest_fixtures_round_trip() -> None:
    fixture_dir = _backtest_fixture_dir()
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fixture_manifest = manifest["fixtures"]
    sessions = {"pre", "regular", "post", "overnight"}

    for name, metadata in fixture_manifest.items():
        path = fixture_dir / name
        assert path.exists()

        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        assert checksum == metadata["sha256"]

        records = list(read_portfolio_file(path))
        assert len(records) == metadata["rows"]

        open_timestamps = [record.open_timestamp for record in records]
        assert open_timestamps == sorted(open_timestamps)

        for record in records:
            assert record.premium is not None
            assert isinstance(record.premium, float)
            assert record.open_timestamp <= record.close_timestamp
            assert record.open_session in sessions
            assert record.close_session in sessions
            assert record.context["source"] == name
            assert record.strategy
            assert isinstance(record.contracts, float)

            if "dte" in name.lower():
                assert record.tenor is not None
                assert record.tenor in record.context.values()
            else:
                assert record.tenor is None
