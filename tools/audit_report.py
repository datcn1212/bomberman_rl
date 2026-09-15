"""Check that every number quoted in the report is still what the data says.

A log this long accumulates numbers that were correct when written and then
stopped being correct - a run was repeated, a conclusion was re-graded, a
configuration changed underneath. This re-derives the headline figures and
compares them against the report text, so a stale number is found by a script
rather than by a reader.

Figures come from `experiments/registry.csv`, which is committed alongside the
models. An earlier version read the per-run logs instead; those live under
`results/`, which is not tracked, so every check silently turned into a SKIP and
the tool reported success while verifying nothing. A check that cannot run is
now a failure, not a pass.

An experiment id is not enough to identify a measurement: the same models are
often measured both 1v1 and on the four-agent board, and those rows sit side by
side in the registry. Each claim therefore names the number of opponents it was
measured against.

    python3 tools/audit_report.py
"""

import csv
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report_tabular_q.md"
REGISTRY = ROOT / "experiments" / "registry.csv"

# (label, experiment id, opponents on the board, metric, value quoted)
CLAIMS = [
    ("Phase 10 gamma 0.9",              "p9_direct",       0, "mean_coins", 1.54),
    ("Phase 10 gamma 0.99",             "p10_g99",         0, "mean_coins", 4.86),
    ("Phase 12 symmetry off, solo",     "p11_g995",        0, "mean_coins", 5.91),
    ("Phase 12 symmetry on, solo",      "p12_sym",         0, "mean_coins", 8.88),
    ("Phase 14 opponents invisible",    "p14_opp_blind",   3, "mean_score", 1.633),
    ("Phase 16 opponents block",        "p16_opp_block",   3, "mean_score", 2.056),
    ("Phase 16 models, 1v1",            "p16_opp_block",   1, "mean_score", 2.919),
    ("Phase 17a all-opponent",          "p17_all_opp",     3, "mean_score", 2.492),
    ("Phase 17b death -15",             "p17_death15",     3, "mean_score", 1.559),
    ("Phase 18 symmetry off, 4-agent",  "p18b_nosym",      3, "mean_score", 2.036),
    ("Phase 19 half budget",            "p19b_6000ep",     3, "mean_score", 2.379),
    ("Phase 20 bomb_safety",            "p20b_bombsafety", 3, "mean_score", 1.875),
    ("Phase 23 the bundle, 1v1",        "p23_replicate",   1, "mean_score", 3.727),
    ("Phase 23 the bundle, crates",     "p23_replicate",   1, "mean_crates", 62.4),
    ("Phase 24 the bundle, 4-agent",    "p23_four",        3, "mean_score", 2.264),
    ("final model, ranking block",      "final_tabular_q", 3, "mean_score", 2.221),
    ("Phase 26 anchor, seeded",         "p26_anchor",      3, "mean_score", 2.210),
    ("Phase 26 anchor, stock",          "p26_anchor_stock", 3, "mean_score", 2.216),
    ("Phase 27 n=3",                    "p27_n3",          3, "mean_score", 1.677),
    ("Phase 27 n=5",                    "p27_n5",          3, "mean_score", 0.985),
    ("Phase 27 n=8",                    "p27_n8",          3, "mean_score", 0.053),
    ("Phase 27 lambda=0.8",             "p27_lam080",      3, "mean_score", 2.159),
    ("Phase 27 lambda=0.9",             "p27_lam090",      3, "mean_score", 1.857),
]

# The three runs of one identical configuration that size the noise floor.
REPEATS = ["p19b_6000ep", "final_tabular_q", "repeat_c"]

TOLERANCE = 0.05


def load_registry():
    if not REGISTRY.exists():
        return []
    with open(REGISTRY) as fh:
        return list(csv.DictReader(fh))


def per_seed(rows, exp_id, opponents, key):
    """Every seed's value for one experiment measured against `opponents` foes.

    Later rows win: re-measuring an experiment appends rather than overwrites,
    and the most recent measurement is the one the report should be quoting.
    """
    latest = {}
    for row in rows:
        if not re.fullmatch(re.escape(exp_id) + r"_seed\d+", row["exp_id"]):
            continue
        # Counts both `rule_based_agent` and `rule_based_seeded`: from Phase 26
        # the opponent is the seeded variant, and the two were measured to be
        # interchangeable (paired t = -0.13), so they index the same setting.
        if row["opponents"].count("rule_based") != opponents:
            continue
        if row[key] in ("", "nan"):
            continue
        latest[row["exp_id"]] = float(row[key])
    return list(latest.values()) or None


def main():
    report = REPORT.read_text()
    rows = load_registry()
    failures = 0
    unchecked = 0

    if not rows:
        print("registry.csv is missing - nothing can be verified")
        return 1

    print("quoted figures against experiments/registry.csv:\n")
    for label, exp_id, opponents, key, quoted in CLAIMS:
        values = per_seed(rows, exp_id, opponents, key)
        if values is None:
            unchecked += 1
            print("  [MISSING] %-30s no rows for %s vs %d opponents"
                  % (label, exp_id, opponents))
            continue
        actual = statistics.mean(values)
        ok = abs(actual - quoted) < TOLERANCE
        failures += not ok
        print("  [%s] %-30s report %7.3f   data %7.3f   (%d seeds)"
              % ("OK   " if ok else "STALE", label, quoted, actual, len(values)))

    print("\nnoise floor from repeated identical runs:")
    means = []
    for name in REPEATS:
        values = per_seed(rows, name, 3, "mean_score")
        if values:
            means.append(statistics.mean(values))
    if len(means) >= 3:
        sd = statistics.stdev(means)
        print("  runs %s" % ", ".join("%.3f" % v for v in means))
        print("  sd %.3f, null standard error for a single comparison %.3f"
              % (sd, sd * math.sqrt(2)))
        quoted_sd = re.search(r"standard\s*\n?deviation of ([\d.]+)", report)
        if quoted_sd:
            ok = abs(float(quoted_sd.group(1)) - sd) < 0.005
            failures += not ok
            print("  [%s] report quotes %s"
                  % ("OK   " if ok else "STALE", quoted_sd.group(1)))
        else:
            unchecked += 1
            print("  [MISSING] the report no longer quotes a run-to-run sd")
    else:
        unchecked += 1
        print("  [MISSING] fewer than three repeat runs in the registry")

    print("\nphase table entries: %d"
          % len(re.findall(r"^\| \d+[ab]? \|", report, re.M)))

    print()
    if failures:
        print("%d FIGURE(S) STALE - fix the report" % failures)
    elif unchecked:
        print("%d CHECK(S) COULD NOT RUN - the rest match, but this is not a pass"
              % unchecked)
    else:
        print("ALL %d QUOTED FIGURES MATCH" % (len(CLAIMS) + 1))
    return 1 if (failures or unchecked) else 0


if __name__ == "__main__":
    sys.exit(main())
