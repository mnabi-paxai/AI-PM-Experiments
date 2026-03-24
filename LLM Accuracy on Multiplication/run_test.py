"""
run_test.py — Test LLM multiplication accuracy vs. Python

For each test config, this script:
  1. Generates random pairs of N-digit numbers
  2. Asks Claude (via AWS Bedrock) to multiply them
  3. Compares the LLM answer to the correct Python answer
  4. Saves detailed results to a CSV file
  5. Prints an accuracy summary

To add a new digit size, just add an entry to TEST_CONFIGS.

Usage:
    python3 run_test.py
"""

import os
import csv
import re
import random
import time
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Add or remove entries here to test different digit sizes
TEST_CONFIGS = [
    {"digits": 2, "samples": 100},
    {"digits": 5, "samples": 100},
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DELAY_BETWEEN_CALLS = 0.3  # seconds — avoids hitting API rate limits


# ── Number Generation ─────────────────────────────────────────────────────────

def random_n_digit_number(digits: int) -> int:
    """Return a random number with exactly `digits` digits."""
    low  = 10 ** (digits - 1)       # e.g. digits=2 → low=10
    high = (10 ** digits) - 1       # e.g. digits=2 → high=99
    return random.randint(low, high)


# ── LLM Interaction ───────────────────────────────────────────────────────────

def ask_llm(client: AnthropicBedrock, num1: int, num2: int) -> tuple[str, int | None]:
    """
    Ask Claude to multiply two numbers.

    Returns:
        raw_response  — the full text Claude returned
        parsed_answer — the integer we extracted from it, or None if parsing failed
    """
    prompt = (
        f"What is {num1} × {num2}? "
        "Respond with only the final number and nothing else. "
        "No explanation, no working, no punctuation — just the number."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    parsed = parse_number(raw)
    return raw, parsed


def parse_number(text: str) -> int | None:
    """
    Extract an integer from the LLM's response.

    Handles cases like:
      "12345"        → 12345
      "12,345"       → 12345  (commas)
      "The answer is 12345."  → 12345
      "≈ 12345"      → 12345
    Returns None if no number can be found.
    """
    cleaned = text.replace(",", "").replace(".", "").strip()
    match = re.search(r"\d+", cleaned)
    return int(match.group()) if match else None


# ── Single Test Run ───────────────────────────────────────────────────────────

def run_test(config: dict, client: AnthropicBedrock) -> dict:
    """
    Run one full test for a given digit size and sample count.

    Returns a summary dict with accuracy stats.
    """
    digits  = config["digits"]
    samples = config["samples"]
    low     = 10 ** (digits - 1)
    high    = (10 ** digits) - 1

    print(f"\n{'='*60}")
    print(f"TEST: {digits}-digit multiplication  ({low}–{high})")
    print(f"Samples: {samples}")
    print(f"{'='*60}")

    # Prepare results CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"results_{digits}digit.csv")

    correct_count = 0
    parse_failures = 0
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample",
            "num1",
            "num2",
            "correct_answer",
            "llm_answer",
            "llm_raw_response",
            "is_correct",
        ])

        for i in range(1, samples + 1):
            num1 = random_n_digit_number(digits)
            num2 = random_n_digit_number(digits)
            correct = num1 * num2     # Python's answer — always right

            try:
                raw, llm_answer = ask_llm(client, num1, num2)
            except Exception as e:
                print(f"  [{i}/{samples}] API error: {e}")
                raw, llm_answer = str(e), None

            is_correct = (llm_answer == correct)

            if llm_answer is None:
                parse_failures += 1
            elif is_correct:
                correct_count += 1

            # Progress indicator
            status = "✓" if is_correct else "✗"
            if llm_answer is None:
                status = "?"
            print(
                f"  [{i:>3}/{samples}]  {num1} × {num2} = {correct}"
                f"  →  LLM: {llm_answer}  {status}"
            )

            writer.writerow([i, num1, num2, correct, llm_answer, raw, is_correct])
            rows.append({
                "num1": num1, "num2": num2,
                "correct": correct, "llm": llm_answer, "ok": is_correct
            })

            time.sleep(DELAY_BETWEEN_CALLS)

    accuracy = (correct_count / samples) * 100

    summary = {
        "digits":         digits,
        "samples":        samples,
        "range":          f"{low}–{high}",
        "correct":        correct_count,
        "wrong":          samples - correct_count - parse_failures,
        "parse_failures": parse_failures,
        "accuracy":       accuracy,
        "csv_path":       csv_path,
    }

    print(f"\nResult: {correct_count}/{samples} correct  →  accuracy = {accuracy:.1f}%")
    if parse_failures:
        print(f"  ({parse_failures} responses could not be parsed as a number)")
    print(f"Saved to: {csv_path}")

    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = AnthropicBedrock()
    summaries = []

    for config in TEST_CONFIGS:
        summary = run_test(config, client)
        summaries.append(summary)

    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<25} {'Samples':>8} {'Correct':>8} {'Wrong':>8} {'Accuracy':>10}")
    print(f"{'-'*60}")
    for s in summaries:
        label = f"{s['digits']}-digit ({s['range']})"
        print(
            f"{label:<25} {s['samples']:>8} {s['correct']:>8} "
            f"{s['wrong']:>8} {s['accuracy']:>9.1f}%"
        )
    print(f"{'='*60}")
    print("\nDetailed results saved to the results/ folder.")
