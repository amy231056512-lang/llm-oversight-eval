"""

RQ2: for the cases where model_a and model_b disagreed at baseline (RQ1),
let them see each other's *structured* judgment (judgment/evidence, not
free-text reasoning), then each gives a final answer. This tests whether
honest disagreement, resolved through a single round of deliberation,
moves each model's judgment toward or away from the ground truth.

Design (single round, not multi-turn -- keeps cost and complexity bounded,
consistent with prior single-turn debate setups in the literature, e.g.
Du et al.'s original multi-agent debate framework and the CW-POR study):

  1. Load RQ1 results for both models (includes evidence, not just a
     verdict).
  2. Restrict to cases where model_a's verdict != model_b's verdict.
  3. For each such case, show model_a its own original judgment alongside
     model_b's structured evidence list (labeled "another reviewer", not
     by model name -- we're not testing brand-recognition effects here),
     and ask for a final structured answer. Do the same symmetrically for
     model_b.
  4. Score two things:
       a. Did each model's final verdict move toward the ground truth
          (correction), away from it (corruption), or stay the same?
       b. Independently, was the *other* model's evidence actually
          accurate (did it reference the real bug_location)? This lets us
          separate "was persuaded by accurate evidence" from "was
          persuaded by evidence that happened to be wrong" -- the latter
          is functionally identical to an unintentional misleading-input
          case, without us having designed one.

Usage:
    export GROQ_API_KEY=gsk_...
    python src/debate.py
"""

import json
from pathlib import Path
from typing import Optional

from baseline import (
    MODELS, DATASET_PATH, RESULTS_DIR,
    add_line_numbers, call_with_retry,
    parse_structured_response, judgment_to_verdict, extract_line_numbers_from_evidence,
)
import time

DISAGREEMENT_PATH = RESULTS_DIR / "disagreement_case_ids.json"
RQ2_RESULTS_DIR = Path("results/debate_results")

DEBATE_PROMPT_TEMPLATE = """You previously reviewed this code and judged it as "{own_judgment}"
(reported bug location: {own_lines}), citing this evidence:
{own_evidence_list}

Another reviewer independently judged the same code as "{other_judgment}"
(reported bug location: {other_lines}), citing this evidence:
{other_evidence_list}

Reconsider the code below in light of the other reviewer's evidence and
reported location. You may keep your original judgment and location or
change either -- base your final answer only on which evidence actually
holds up under scrutiny, not on simply deferring to the other reviewer.

Specification:
{spec}

Code (line numbers shown for reference):
```python
{numbered_code}
```

Respond with ONLY a single JSON object, no other text before or after it,
in exactly this shape:
{{
  "judgment": "BUG" or "OK",
  "line_numbers": [<line number(s) where the bug is -- REQUIRED and must be
                    non-empty if judgment is "BUG"; empty list if judgment
                    is "OK">],
  "evidence": ["<short claim>", "<short claim>", ...]
}}
"""


def load_rq1_results(label: str, model_name: str) -> dict:
    path = RESULTS_DIR / f"rq1_{label}_{model_name.replace('/', '_')}.jsonl"
    with open(path) as f:
        return {(rec := json.loads(line))["id"]: rec for line in f}


def load_cases_by_id() -> dict:
    with open(DATASET_PATH) as f:
        return {(rec := json.loads(line))["id"]: rec for line in f}


def format_evidence_list(evidence: list) -> str:
    if not evidence:
        return "  (no evidence given)"
    return "\n".join(f"  - {item}" for item in evidence)


def format_lines(line_numbers: list) -> str:
    if not line_numbers:
        return "none given"
    return ", ".join(str(n) for n in sorted(line_numbers))


def evidence_matches_true_bug(evidence: list, true_lines: list) -> bool:
    """Check whether a set of evidence strings actually references the real
    bug line(s) -- used to tell 'persuaded by accurate evidence' apart from
    'persuaded by evidence that was itself wrong', for cases where the
    ground truth is 'incorrect' (i.e. there is a real bug to point at)."""
    if not true_lines:
        return False
    reported = extract_line_numbers_from_evidence(evidence)
    return any(n in true_lines for n in reported)


def run_debate_round(disagreement_ids: list, cases_by_id: dict,
                      a_results: dict, b_results: dict,
                      disagreement_type: dict) -> list:
    debate_results = []
    for i, case_id in enumerate(disagreement_ids):
        case = cases_by_id[case_id]
        numbered_code = add_line_numbers(case["code"])
        true_lines = case.get("bug_location", {}).get("line_numbers", [])

        a_initial, b_initial = a_results[case_id], b_results[case_id]

        # model_a sees model_b's structured judgment + evidence + location
        prompt_for_a = DEBATE_PROMPT_TEMPLATE.format(
            own_judgment=a_initial["verdict"],
            own_lines=format_lines(a_initial.get("reported_lines", [])),
            own_evidence_list=format_evidence_list(a_initial.get("evidence", [])),
            other_judgment=b_initial["verdict"],
            other_lines=format_lines(b_initial.get("reported_lines", [])),
            other_evidence_list=format_evidence_list(b_initial.get("evidence", [])),
            spec=case["spec"], numbered_code=numbered_code,
        )
        a_response, _ = call_with_retry(MODELS["model_a"], prompt_for_a)
        a_parsed = parse_structured_response(a_response)
        a_final_verdict = judgment_to_verdict(a_parsed["judgment"])

        time.sleep(1)

        # model_b sees model_a's structured judgment + evidence + location
        prompt_for_b = DEBATE_PROMPT_TEMPLATE.format(
            own_judgment=b_initial["verdict"],
            own_lines=format_lines(b_initial.get("reported_lines", [])),
            own_evidence_list=format_evidence_list(b_initial.get("evidence", [])),
            other_judgment=a_initial["verdict"],
            other_lines=format_lines(a_initial.get("reported_lines", [])),
            other_evidence_list=format_evidence_list(a_initial.get("evidence", [])),
            spec=case["spec"], numbered_code=numbered_code,
        )
        b_response, _ = call_with_retry(MODELS["model_b"], prompt_for_b)
        b_parsed = parse_structured_response(b_response)
        b_final_verdict = judgment_to_verdict(b_parsed["judgment"])

        def classify_shift(initial_correct: bool, final_correct: bool) -> str:
            if initial_correct and not final_correct:
                return "corrupted"     # was right, deliberation made it wrong
            if not initial_correct and final_correct:
                return "corrected"     # was wrong, deliberation made it right
            if initial_correct and final_correct:
                return "stayed_correct"
            return "stayed_incorrect"

        def classify_location_shift(initial_lines: list, final_lines: list) -> Optional[str]:
            """Same shift logic as classify_shift, but for whether reported
            lines actually overlap the real bug location -- only meaningful
            when there's a real bug to locate (ground_truth == 'incorrect')
            and the final verdict is still 'incorrect' (no location claim
            to score once a model says 'OK')."""
            if case["ground_truth"] != "incorrect" or not true_lines:
                return None
            initial_correct = any(n in true_lines for n in initial_lines)
            final_correct = any(n in true_lines for n in final_lines)
            if initial_correct and not final_correct:
                return "corrupted"
            if not initial_correct and final_correct:
                return "corrected"
            if initial_correct and final_correct:
                return "stayed_correct"
            return "stayed_incorrect"

        a_initial_correct = a_initial["verdict"] == case["ground_truth"]
        a_final_correct = a_final_verdict == case["ground_truth"]
        b_initial_correct = b_initial["verdict"] == case["ground_truth"]
        b_final_correct = b_final_verdict == case["ground_truth"]

        a_final_lines = a_parsed["line_numbers"] if a_final_verdict == "incorrect" else []
        b_final_lines = b_parsed["line_numbers"] if b_final_verdict == "incorrect" else []
        a_location_shift = classify_location_shift(
            a_initial.get("reported_lines", []) if a_initial["verdict"] == "incorrect" else [],
            a_final_lines,
        )
        b_location_shift = classify_location_shift(
            b_initial.get("reported_lines", []) if b_initial["verdict"] == "incorrect" else [],
            b_final_lines,
        )

        # Was the evidence each model was *shown* actually accurate? Only
        # meaningful when there's a real bug to point at.
        b_evidence_was_accurate = (
            evidence_matches_true_bug(b_initial.get("evidence", []), true_lines)
            if case["ground_truth"] == "incorrect" else None
        )
        a_evidence_was_accurate = (
            evidence_matches_true_bug(a_initial.get("evidence", []), true_lines)
            if case["ground_truth"] == "incorrect" else None
        )

        debate_results.append({
            "id": case_id,
            "disagreement_type": disagreement_type.get(case_id, "unknown"),
            "ground_truth": case["ground_truth"],
            "model_a_initial": a_initial["verdict"],
            "model_a_final": a_final_verdict,
            "model_a_shift": classify_shift(a_initial_correct, a_final_correct),
            "model_a_initial_lines": a_initial.get("reported_lines", []),
            "model_a_final_lines": a_final_lines,
            "model_a_location_shift": a_location_shift,
            "model_a_final_evidence": a_parsed["evidence"],
            "model_b_initial": b_initial["verdict"],
            "model_b_final": b_final_verdict,
            "model_b_shift": classify_shift(b_initial_correct, b_final_correct),
            "model_b_initial_lines": b_initial.get("reported_lines", []),
            "model_b_final_lines": b_final_lines,
            "model_b_location_shift": b_location_shift,
            "model_b_final_evidence": b_parsed["evidence"],
            # was the evidence *shown to the other model* actually accurate?
            "evidence_a_was_accurate": a_evidence_was_accurate,  # shown to model_b
            "evidence_b_was_accurate": b_evidence_was_accurate,  # shown to model_a
            "model_a_raw_response": a_response.strip(),
            "model_b_raw_response": b_response.strip(),
        })

        print(f"  [{i+1}/{len(disagreement_ids)}] {case_id}  (gt={case['ground_truth']}): "
              f"A {a_initial['verdict']}→{a_final_verdict} ({debate_results[-1]['model_a_shift']})  |  "
              f"B {b_initial['verdict']}→{b_final_verdict} ({debate_results[-1]['model_b_shift']})")
        time.sleep(1)

    return debate_results


def summarize(debate_results: list):
    total = len(debate_results)
    print(f"\n=== RQ2 Deliberation — {total} disagreement cases ===")

    for model_key in ["model_a", "model_b"]:
        shifts = [r[f"{model_key}_shift"] for r in debate_results]
        n_corrected = shifts.count("corrected")
        n_corrupted = shifts.count("corrupted")
        n_stayed_correct = shifts.count("stayed_correct")
        n_stayed_incorrect = shifts.count("stayed_incorrect")
        print(f"\n{model_key} ({MODELS[model_key]}):")
        print(f"  Corrected (wrong -> right): {n_corrected}/{total}")
        print(f"  Corrupted (right -> wrong): {n_corrupted}/{total}")
        print(f"  Stayed correct:             {n_stayed_correct}/{total}")
        print(f"  Stayed incorrect:           {n_stayed_incorrect}/{total}")

    before_correct = sum(
        (r["model_a_initial"] == r["ground_truth"]) + (r["model_b_initial"] == r["ground_truth"])
        for r in debate_results
    )
    after_correct = sum(
        (r["model_a_final"] == r["ground_truth"]) + (r["model_b_final"] == r["ground_truth"])
        for r in debate_results
    )
    n_judgments = total * 2
    print(f"\nCombined accuracy on disagreement cases:")
    print(f"  Before deliberation: {before_correct}/{n_judgments} ({before_correct/n_judgments:.0%})")
    print(f"  After deliberation:  {after_correct}/{n_judgments} ({after_correct/n_judgments:.0%})")
    if after_correct > before_correct:
        print("  -> Net effect: deliberation IMPROVED accuracy on disagreement cases.")
    elif after_correct < before_correct:
        print("  -> Net effect: deliberation DEGRADED accuracy on disagreement cases.")
    else:
        print("  -> Net effect: no net change.")

    # This is the analysis that distinguishes B from a designed misleading-
    # assistance study: does being shown *inaccurate* evidence explain the
    # corrupted/corrected outcomes?
    print("\n=== Evidence accuracy vs. outcome (bridges to the 'misleading' framing) ===")
    corrupted_with_bad_evidence = sum(
        1 for r in debate_results
        if r["model_a_shift"] == "corrupted" and r["evidence_b_was_accurate"] is False
    ) + sum(
        1 for r in debate_results
        if r["model_b_shift"] == "corrupted" and r["evidence_a_was_accurate"] is False
    )
    corrected_with_good_evidence = sum(
        1 for r in debate_results
        if r["model_a_shift"] == "corrected" and r["evidence_b_was_accurate"] is True
    ) + sum(
        1 for r in debate_results
        if r["model_b_shift"] == "corrected" and r["evidence_a_was_accurate"] is True
    )
    print(f"Corrupted judgments where the persuading evidence was itself inaccurate: "
          f"{corrupted_with_bad_evidence}")
    print(f"Corrected judgments where the persuading evidence was accurate: "
          f"{corrected_with_good_evidence}")


def main():
    if not DISAGREEMENT_PATH.exists():
        print(f"'{DISAGREEMENT_PATH}' not found -- run baseline.py first.")
        return

    with open(DISAGREEMENT_PATH) as f:
        disagreement_data = json.load(f)

    # baseline.py saves a dict with verdict/location disagreements tracked
    # separately: {"verdict_disagreements": [...], "location_disagreements":
    # [...], "all": [...]}. Build a case_id -> type lookup so each debate
    # result records which kind of disagreement brought it into this round
    # (a verdict flip is a different, more basic disagreement than two
    # models agreeing "there's a bug" but pointing at different lines).
    disagreement_ids = disagreement_data.get("all", [])
    disagreement_type = {}
    for cid in disagreement_data.get("verdict_disagreements", []):
        disagreement_type[cid] = "verdict"
    for cid in disagreement_data.get("location_disagreements", []):
        disagreement_type[cid] = "location"

    if not disagreement_ids:
        print("No disagreement cases found between model_a and model_b at baseline -- "
              "RQ2 as designed has nothing to test. Consider adding more cases or "
              "checking whether the two models are too similar in judgment.")
        return

    print(f"Loaded {len(disagreement_ids)} disagreement cases "
          f"({len(disagreement_data.get('verdict_disagreements', []))} verdict, "
          f"{len(disagreement_data.get('location_disagreements', []))} location)")

    cases_by_id = load_cases_by_id()
    a_results = load_rq1_results("model_a", MODELS["model_a"])
    b_results = load_rq1_results("model_b", MODELS["model_b"])

    debate_results = run_debate_round(disagreement_ids, cases_by_id, a_results, b_results,
                                       disagreement_type)
    summarize(debate_results)

    RQ2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RQ2_RESULTS_DIR / "rq2_debate_results.jsonl"
    with open(out_path, "w") as f:
        for r in debate_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()