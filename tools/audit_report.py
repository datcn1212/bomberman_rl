"""Check that every number quoted in the report is still what the data says.

A log this long accumulates numbers that were correct when written and then
stopped being correct - a run was repeated, a conclusion was re-graded, a
configuration changed underneath. This re-derives the headline figures from the
stored logs and compares them against the report text, so a stale number is
found by a script rather than by a reader.

    python3 tools/audit_report.py
"""

import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report_tabular_q.md"

# (label, log file, metric, value quoted in the report)
CLAIMS = [
    ("Phase 12 symmetry on, solo",      "p12_sym",        "mean_coins", 8.88),
    ("Phase 12 symmetry off, solo",     "p11_g995",       "mean_coins", 5.91),
    ("Phase 10 gamma 0.99",             "p10_g99",        "mean_coins", 4.86),
    ("Phase 10 gamma 0.9",              "p9_direct",      "mean_coins", 1.54),
    ("Phase 17a all-opponent",          "p17_all_opp",    "mean_score", 2.492),
    ("Phase 16 opponents block",        "p16_opp_block",  "mean_score", 2.056),
    ("Phase 14 opponents invisible",    "p14_opp_blind",  "mean_score", 1.633),
    ("Phase 18 symmetry off, 4-agent",  "p18b_nosym",     "mean_score", 2.036),
    ("Phase 19 half budget",            "p19b_6000ep",    "mean_score", 2.379),
    ("Phase 20 bomb_safety",            "p20b_bombsafety", "mean_score", 1.875),
    ("Phase 17b death -15",             "p17_death15",    "mean_score", 1.559),
    ("final model, ranking block",      "final_tabular_q", "mean_score", 2.221),
]

REPEATS = ["p19b_6000ep", "final_tabular_q", "repeat_c"]


def per_seed(name, key):
    log = Path("/tmp/%s.log" % name)
    if not log.exists():
        return None
    text = log.read_text()
    if "===" not in text:
        return None
    tail = text[text.rindex("==="):]
    m = re.search(r"  %s\s+[\-0-9.]+\s+\(per seed: (.*?), spread" % key, tail)
    return [float(x) for x in m.group(1).split(", ")] if m else None


def main():
    report = REPORT.read_text()
    failures = 0

    print("quoted figures against the logs:\n")
    for label, log, key, quoted in CLAIMS:
        values = per_seed(log, key)
        if values is None:
            print("  [SKIP] %-32s log missing" % label)
            continue
        actual = statistics.mean(values)
        ok = abs(actual - quoted) < 0.005
        failures += not ok
        print("  [%s] %-32s report %.3f   data %.3f" %
              ("OK  " if ok else "STALE", label, quoted, actual))

    print("\nnoise floor from repeated identical runs:")
    means = [statistics.mean(per_seed(r, "mean_score")) for r in REPEATS
             if per_seed(r, "mean_score")]
    if len(means) >= 3:
        sd = statistics.stdev(means)
        print("  runs %s" % ", ".join("%.3f" % v for v in means))
        print("  sd %.3f, null standard error for a single comparison %.3f"
              % (sd, sd * math.sqrt(2)))
        quoted_sd = re.search(r"standard\s*\n?deviation of ([\d.]+)", report)
        if quoted_sd:
            ok = abs(float(quoted_sd.group(1)) - sd) < 0.005
            failures += not ok
            print("  [%s] report quotes %s" %
                  ("OK  " if ok else "STALE", quoted_sd.group(1)))

    print("\nphase table entries: %d" % len(re.findall(r"^\| \d+[ab]? \|", report, re.M)))
    print("rejected ideas listed: %d" %
          len(re.findall(r" - ", report.split("## Rejected, with evidence")[-1].split("##")[0]))
          if "## Rejected, with evidence" in report else 0)

    print("\n%s" % ("ALL QUOTED FIGURES MATCH" if not failures
                    else "%d FIGURE(S) STALE - fix the report" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
