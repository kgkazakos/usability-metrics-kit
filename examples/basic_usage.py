"""Minimal end-to-end example for usability-metrics-kit.

Run:  python examples/basic_usage.py
"""

from __future__ import annotations

from usability_metrics_kit import (
    composite_score,
    net_promoter_score,
    score_sus,
    task_success_rate,
    time_on_task,
)

# Eight respondents answered the 10-item SUS (1-5 each).
sus_responses = [
    [4, 2, 4, 2, 4, 1, 4, 2, 4, 2],
    [5, 1, 4, 2, 5, 1, 4, 1, 4, 2],
    [4, 2, 3, 3, 4, 2, 4, 2, 3, 2],
    [5, 2, 5, 1, 4, 1, 5, 1, 5, 1],
    [3, 3, 4, 2, 3, 2, 4, 3, 4, 2],
    [4, 1, 4, 2, 5, 1, 4, 2, 4, 1],
    [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
    [3, 2, 3, 3, 4, 2, 3, 2, 4, 2],
]

sus = score_sus(sus_responses)
print(f"SUS: {sus.score}  ({sus.benchmark['grade']}, {sus.benchmark['adjective']})")
print(f"     95% CI {sus.interval.low}–{sus.interval.high}")

success = task_success_rate([1, 1, 1, 0, 1, 1, 0, 1])
print(f"Completion: {success.score}%  CI {success.interval.low}–{success.interval.high}")

times = time_on_task([42, 38, 55, 61, 29, 47, 88, 40])
print(f"Time on task (geo mean): {times.score}s")

nps = net_promoter_score([10, 9, 9, 8, 7, 6, 10, 9])
print(f"NPS: {nps.score}  ({nps.benchmark['band']})")

# Roll a few sub-scores into one directional composite.
comp = composite_score(
    {
        "satisfaction": sus.score,
        "completion": success.score,
        "loyalty": (nps.score + 100) / 2,  # map -100..100 onto 0..100
    }
)
print(f"Usability composite: {comp.score}/100")
