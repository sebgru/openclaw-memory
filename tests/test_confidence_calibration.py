import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "confidence_calibration", Path(__file__).parents[1] / "scripts" / "confidence_calibration.py"
)
cc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cc)


def rec(sim, relevant, lexical=0, exact=False):
    return {
        "semantic_similarity": sim,
        "relevant": relevant,
        "lexical_evidence": lexical,
        "exact_match": exact,
    }


def test_wilson_interval_known_values():
    assert cc.wilson_interval(0, 0) == (None, None)
    low, high = cc.wilson_interval(5, 10)
    assert round(low, 3) == 0.237 and round(high, 3) == 0.763
    low, high = cc.wilson_interval(0, 5)
    assert low == 0.0 and 0.43 < high < 0.44


def test_calibrate_splits_at_floor_and_ignores_lexical_hits():
    records = [
        rec(0.50, False),
        rec(0.55, False),
        rec(0.59, True),
        rec(0.60, True),
        rec(0.70, True),
        rec(0.30, False, lexical=4.0),
        rec(0.30, False, exact=True),
    ]
    report = cc.calibrate(records, floor=0.6)
    assert report["n"] == 5
    assert report["below_floor"]["n"] == 3 and report["below_floor"]["relevant"] == 1
    assert report["at_or_above_floor"]["n"] == 2
    assert report["at_or_above_floor"]["relevant_rate"] == 1.0
    assert report["irrelevant_flagged_rate"] == 1.0
    assert abs(report["relevant_flagged_rate"] - 1 / 3) < 1e-9
    assert cc.calibrate(records, 0.6, lexical_free_only=False)["n"] == 7


def test_empty_and_cli(tmp_path, capsys):
    assert cc.calibrate([])["below_floor"]["relevant_rate"] is None
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(rec(0.4, False)) + "\n\n" + json.dumps(rec(0.8, True)) + "\n")
    assert cc.main([str(path), "--floor", "0.6"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 2
