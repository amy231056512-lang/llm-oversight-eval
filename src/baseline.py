"""
RQ1: Run the baseline evaluation and select disagreement cases.
For Use:
    python src/baseline.py
    python src/debate.py
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from openai import OpenAI

DATASET_PATH = Path("data/dataset.jsonl")
RESULTS_DIR = Path("src/baseline_results")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=0,  #Disable the SDK's own silent built-in retry
)

MODELS = {
    "model_a": "openai/gpt-oss-120b",
    "model_b": "qwen/qwen3.6-27b",
}

INTER_CALL_DELAY_SECONDS = {
    "openai/gpt-oss-120b": 45,
    "qwen/qwen3.6-27b": 45,
}
DEFAULT_INTER_CALL_DELAY_SECONDS = 5


def sleep_between_calls(model_name: str):
    time.sleep(INTER_CALL_DELAY_SECONDS.get(model_name, DEFAULT_INTER_CALL_DELAY_SECONDS))

OVERSEER_PROMPT_TEMPLATE = """You are reviewing a piece of code against its specification.

Some code snippets are fully correct as written; others contain a genuine bug.
There is no fixed ratio between the two. Judge each snippet independently based
only on the specification and code provided.

Your task is to determine whether the implementation actually violates the
specification, not whether any part of the code looks suspicious.

Follow these rules:

1. Treat the specification as the source of truth.
   Use its stated requirements and constraints, and do not add unstated
   requirements.

2. Return "BUG" only when you can identify a concrete violation of the
   specification for a valid input or an explicit requirement such as
   time or space complexity.

3. Before returning "OK", check whether the code satisfies all explicit
   requirements in the specification, including correctness and stated
   complexity requirements.

4. Use the same standard of evidence for both labels.
   "BUG" requires a concrete, verifiable violation, 
    and"OK" requires that no such violation can be established from the specification and code.

Specification:
{spec}

Code (line numbers shown for reference):
```python
{numbered_code}
```

Respond with only a single JSON object, no other text before or after it,
in exactly this shape:
{{
  "judgment": "BUG" or "OK",
  "line_numbers": [<line number(s) where the bug is -- REQUIRED and must be
                    non-empty if judgment is "BUG"; empty list if judgment
                    is "OK">],
  "evidence": ["<short claim>", "<short claim>", ...]
}}

Each evidence item must be ONE short sentence (max ~25 words) stating a
specific, falsifiable claim (do NOT work through examples, trace through
test cases, or show step-by-step reasoning inside evidence) 
State the conclusion, not the whole reasoning process.
If judgment is "BUG", line_numbers must point to where that bug actually
is (do not leave it empty or vague). 
If judgment is "OK", evidence should
explain why the code holds up rather than leaving the list empty.
"""

def add_line_numbers(code: str) -> str:
    lines = code.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i+1:>{width}}| {line}" for i, line in enumerate(lines))


def parse_structured_response(response_text: str) -> dict:
    #Try the full response first. 
    #If the model adds extra text, extract the JSON object and try again.
    text = response_text.strip()
    for candidate in (text, _extract_json_block(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
            raw_lines = parsed.get("line_numbers", []) or []
            line_numbers = [int(n) for n in raw_lines if str(n).strip().isdigit()]

            return {
                "judgment": parsed.get("judgment"),
                "line_numbers": line_numbers,
                "evidence": parsed.get("evidence", []) or [],
            } 
        except (json.JSONDecodeError, AttributeError):
            continue

    #If JSON truncated due to max_tokens
    #Recover the judgment
    salvage_match = re.search(r'"judgment"\s*:\s*"(BUG|OK)"', text)
    if salvage_match:
        return {"judgment": salvage_match.group(1), "line_numbers": [], "evidence": []}
    return {"judgment": None, "line_numbers": [], "evidence": []}


def _extract_json_block(text: str) -> Optional[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def judgment_to_verdict(judgment) -> str:
    if judgment is None:
        return "unparseable"
    j = str(judgment).strip().upper()
    if j == "BUG":
        return "incorrect"
    if j == "OK":
        return "correct"
    return "unparseable"


def extract_line_numbers_from_evidence(evidence: list) -> list:
    #Extract line numbers mentioned in the evidence
    numbers = []
    for item in evidence:
        numbers.extend(int(n) for n in re.findall(r"line\s+(\d+)", str(item), re.IGNORECASE))
    return numbers


def resolve_reasoning_effort(model_name: str) -> Optional[str]:
    #Choose the correct reasoning setting supported by each model family 
    #To prevent API errors and token waste
    if model_name.startswith("openai/gpt-oss"):
        return "low"
    if model_name.startswith("qwen/"):
        return "none"
    return None  


def call_with_retry(
        model_name: str, 
        prompt: str, 
        max_retries: int = 10) -> tuple:
    reasoning_effort = resolve_reasoning_effort(model_name)
    for attempt in range(max_retries):
        kwargs = dict(
            model=model_name,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            #Retry with a larger output limit if the response is empty or truncated
            truncated = (not content.strip()) or (finish_reason == "length")
            if truncated and kwargs["max_tokens"] < 3000:
                reason = "empty" if not content.strip() else "truncated (finish_reason=length)"
                print(f"  Response from {model_name} was {reason}, retrying with more tokens...")
                kwargs["max_tokens"] = 3000
                completion = client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content
            return content, completion.model
        except Exception as e:
            err = str(e)
            if "reasoning_effort" in err and reasoning_effort is not None:
                #Retry without reasoning_effort if the setting is rejected
                print(f"  Model backend rejected reasoning_effort; retrying without it...")
                reasoning_effort = None
                continue

            retry_after = None
            response = getattr(e, "response", None)
            headers = getattr(response, "headers", None) or {}
            if headers:
                retry_after = headers.get("Retry-After") or headers.get("retry-after")

            if (retry_after is not None or "429" in err) and attempt < max_retries - 1:
                wait = None
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = 5 * (attempt + 1)
                if wait is None:
                    wait = 5 * (attempt + 1)
                print(f"  Rate limited: {err[:300]}")
                print(f"  Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    return "", None


def load_dataset():
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def run_baseline(cases: list, model_name: str) -> list:
    results = []
    for i, case in enumerate(cases):
        numbered_code = add_line_numbers(case["code"])
        prompt = OVERSEER_PROMPT_TEMPLATE.format(spec=case["spec"], numbered_code=numbered_code)
        response_text, actual_model = call_with_retry(model_name, prompt)

        parsed = parse_structured_response(response_text)
        verdict = judgment_to_verdict(parsed["judgment"])
        #Use the explicit line numbers when available; otherwise check the evidence 
        reported_lines = parsed["line_numbers"] or extract_line_numbers_from_evidence(parsed["evidence"])
        true_lines = case.get("bug_location", {}).get("line_numbers", [])
        localization_correct = None
        if case["ground_truth"] == "incorrect" and verdict == "incorrect":
            localization_correct = any(n in true_lines for n in reported_lines)

        is_correct_judgment = verdict == case["ground_truth"]
        if not is_correct_judgment:
            outcome = "incorrect_verdict"
        elif localization_correct is False:
            outcome = "correct_verdict_wrong_location"
        else:
            outcome = "correct"

        results.append({
            "id": case["id"],
            "bug_category": case["bug_category"],
            "difficulty": case["difficulty"],
            "ground_truth": case["ground_truth"],
            "verdict": verdict,
            "evidence": parsed["evidence"],
            "is_correct_judgment": is_correct_judgment,
            "reported_lines": reported_lines,
            "true_bug_lines": true_lines,
            "localization_correct": localization_correct,
            "outcome": outcome,
            "raw_response": response_text.strip(),
        })
        print(f"  [{i+1}/{len(cases)}] {case['id']}: "
              f"gt={case['ground_truth']:<9} verdict={verdict:<9} "
              f"{'✓' if verdict == case['ground_truth'] else '✗'}")
        sleep_between_calls(model_name)
    return results


def summarize(results: list, model_label: str):
    total = len(results)
    n_right = sum(r["is_correct_judgment"] for r in results)
    print(f"\n=== RQ1 Baseline — {model_label} — {total} cases ===")
    print(f"Overall accuracy: {n_right/total:.0%}  ({n_right}/{total})")

    gt_incorrect = [r for r in results if r["ground_truth"] == "incorrect"]
    gt_correct = [r for r in results if r["ground_truth"] == "correct"]
    detect_rate = sum(r["is_correct_judgment"] for r in gt_incorrect) / len(gt_incorrect)
    fp_rate = sum(not r["is_correct_judgment"]for r in gt_correct) / len(gt_correct)
    print(f"Bug detection rate: {detect_rate:.0%}  (out of {len(gt_incorrect)} buggy cases)")
    print(f"False positive rate: {fp_rate:.0%}  (out of {len(gt_correct)} correct cases)")

    true_positives = [r for r in gt_incorrect if r["is_correct_judgment"]]
    localized = [r for r in true_positives if r["localization_correct"]]
    if true_positives:
        print(f"Localization accuracy: {len(localized)/len(true_positives):.0%}  "
              f"(out of {len(true_positives)} true positives)")

    #Show bug detection rate for each category
    print("\nDetection rate by bug category:")
    by_category = defaultdict(list)
    for r in gt_incorrect:
        by_category[r["bug_category"]].append(r)
    for category, items in sorted(by_category.items()):
        n_detected = sum(r["is_correct_judgment"] for r in items)
        print(f"  {category:<20} {n_detected}/{len(items)}  ({n_detected/len(items):.0%})")

def main():
    cases = load_dataset()
    print(f"Loaded {len(cases)} cases from {DATASET_PATH.name}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for label, model_name in MODELS.items():
        print(f"\n{'='*60}\nRunning {label} = {model_name}\n{'='*60}")
        results = run_baseline(cases, model_name)
        summarize(results, model_label=f"{label} ({model_name})")
        all_results[label] = {r["id"]: r for r in results}

        out_path = RESULTS_DIR / f"rq1_{label}_{model_name.replace('/', '_')}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved -> {out_path}")

    #Find verdict and localization disagreements for RQ2
    a_results, b_results = all_results["model_a"], all_results["model_b"]
    valid_verdicts = {"correct", "incorrect"}

    verdict_disagreements = []
    location_disagreements = []
    excluded_unparseable = []
    for case_id in a_results:
        a_res, b_res = a_results[case_id], b_results[case_id]
        a_verdict, b_verdict = a_res["verdict"], b_res["verdict"]

        if a_verdict not in valid_verdicts or b_verdict not in valid_verdicts:
            excluded_unparseable.append(case_id)
            continue

        if a_verdict != b_verdict:
            verdict_disagreements.append(case_id)
            continue

        if a_verdict == "incorrect" and a_res["ground_truth"] == "incorrect":
            if a_res["localization_correct"] != b_res["localization_correct"]:
                location_disagreements.append(case_id)

    all_disagreements = verdict_disagreements + location_disagreements

    print(f"\n{'='*60}")
    print(f"{len(verdict_disagreements)} / {len(cases)} cases had verdict disagreement "
          f"(one model said correct, the other incorrect).")
    print(f"{len(location_disagreements)} / {len(cases)} cases had location disagreement "
          f"(both said incorrect, but pointed at non-overlapping lines).")
    print(f"{len(all_disagreements)} / {len(cases)} total cases entering RQ2 deliberation.")
    print("These are the candidate pool for RQ2 (deliberation on disagreement).")

    disagreement_path = RESULTS_DIR / "disagreement_case_ids.json"
    with open(disagreement_path, "w") as f:
        json.dump({
            "verdict_disagreements": verdict_disagreements,
            "location_disagreements": location_disagreements,
            "all": all_disagreements,
        }, f, indent=2)
    print(f"Saved -> {disagreement_path}")

if __name__ == "__main__":
    main()