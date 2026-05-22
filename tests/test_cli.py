"""CLI tests."""

from __future__ import annotations

import json

from click.testing import CliRunner

from usability_metrics_kit.cli import cli


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_list_runs():
    res = CliRunner().invoke(cli, ["list"])
    assert res.exit_code == 0
    assert "sus" in res.output


def test_score_sus_json(tmp_path):
    csv = _write(
        tmp_path, "sus.csv",
        "Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10\n"
        "5,1,5,1,5,1,5,1,5,1\n5,1,5,1,5,1,5,1,5,1\n"
        "5,1,5,1,5,1,5,1,5,1\n5,1,5,1,5,1,5,1,5,1\n5,1,5,1,5,1,5,1,5,1\n",
    )
    res = CliRunner().invoke(cli, ["score", "-m", "sus", csv, "--json-output", "--no-llm"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["score"] == 100.0
    assert data["llm_summary"] is None


def test_score_nps_rich(tmp_path):
    csv = _write(tmp_path, "nps.csv", "score\n10\n9\n8\n7\n6\n")
    res = CliRunner().invoke(cli, ["score", "-m", "nps", csv, "--no-llm"])
    assert res.exit_code == 0
    assert "Net Promoter Score" in res.output


def test_score_bad_shape_exits(tmp_path):
    csv = _write(tmp_path, "bad.csv", "Q1,Q2\n5,1\n")
    res = CliRunner().invoke(cli, ["score", "-m", "sus", csv, "--no-llm"])
    assert res.exit_code == 1
    assert "error" in res.output.lower()


def test_scorecard(tmp_path):
    sus = _write(
        tmp_path, "sus.csv",
        "Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10\n"
        + "\n".join(["4,2,4,2,4,2,4,2,4,2"] * 6) + "\n",
    )
    nps = _write(tmp_path, "nps.csv", "score\n10\n9\n8\n7\n6\n5\n")
    spec = _write(tmp_path, "spec.json", json.dumps({"sus": sus, "nps": nps}))
    res = CliRunner().invoke(
        cli, ["scorecard", spec, "--study", "Demo", "--json-output", "--no-llm"]
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["study"] == "Demo"
    assert {r["key"] for r in data["results"]} == {"sus", "nps"}
