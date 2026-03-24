# LLM Accuracy on Multiplication — v2: Three-Way Comparison

An extended benchmarking experiment comparing three approaches to multiplication across two levels of numerical complexity. The core question: **does giving an LLM access to a calculator tool fix its arithmetic?**

---

## The Three Approaches

| Approach | How it works | Expected accuracy |
|---|---|---|
| **Python** | Direct computation (`num1 * num2`) | 100% — always |
| **LLM Direct** | Claude answers from memory/pattern-matching | High for simple, low for complex |
| **LLM + Tool** | Claude calls a calculator tool, Python executes it, Claude returns the result | Should approach 100% |

The third approach — **LLM as an agent with tool use** — is the central new concept in this experiment.

---

## What Is Tool Use?

Tool use (also called function calling) is a mechanism where an LLM can decide to call an external function instead of answering directly. The model doesn't execute the tool itself — it outputs a structured request saying "I want to call this function with these inputs." Your code executes the function and returns the result. The model then uses that result in its final answer.

```
User: "What is 84732 × 61409?"
         │
         ▼
Claude (Turn 1):
  "I'll use the calculator."
  → tool_use: calculator(num1=84732, num2=61409, operation="multiply")
         │
         ▼
Your code executes: 84732 * 61409 = 5,202,157,788  (Python, exact)
         │
         ▼
Claude (Turn 2):
  Receives: tool_result = "5202157788"
  Returns: "5202157788"
         │
         ▼
Answer: 5,202,157,788  ✓
```

The LLM never does the arithmetic. It just decides *to use* the tool, and delegates the computation to Python. This is the right architecture for any numerical reasoning task.

---

## Why This Matters

This experiment demonstrates a fundamental principle in AI product design:

> **LLMs are reasoning engines, not calculators. Give them tools for the things they're bad at.**

The pattern generalizes far beyond multiplication:

| Task | Don't use LLM alone | Use LLM + Tool |
|---|---|---|
| Arithmetic | LLM predicts digits | Code interpreter |
| Current information | LLM hallucinates | Web search tool |
| Database queries | LLM guesses data | SQL executor |
| File operations | LLM imagines content | File system tool |
| Exact dates/times | LLM confuses dates | Clock/calendar API |

---

## The Agentic Loop

The tool use pattern turns a single LLM call into a **two-turn agentic conversation**:

```
┌─────────────────────────────────────────────────────────────┐
│  TURN 1: LLM decides to use the tool                        │
│                                                             │
│  Input:  user question + tool definition                    │
│  Output: tool_use block {name, id, inputs}                  │
│  stop_reason: "tool_use"                                    │
└─────────────────────────────────────────────────────────────┘
              │
              ▼  your code executes the tool
┌─────────────────────────────────────────────────────────────┐
│  TURN 2: LLM receives the result and answers                │
│                                                             │
│  Input:  original question + tool result                    │
│  Output: final text answer                                  │
│  stop_reason: "end_turn"                                    │
└─────────────────────────────────────────────────────────────┘
```

Each sample in this experiment uses **2 API calls** (vs. 1 for LLM Direct), which is why it takes longer to run.

---

## The Calculator Tool Definition

The tool is described to Claude in JSON schema format:

```python
{
    "name": "calculator",
    "description": "Performs exact arithmetic. Always use this for multiplication.",
    "input_schema": {
        "type": "object",
        "properties": {
            "num1":      {"type": "number"},
            "num2":      {"type": "number"},
            "operation": {"type": "string", "enum": ["multiply"]}
        },
        "required": ["num1", "num2", "operation"]
    }
}
```

Claude reads this description and decides when to call it. The description is important — "Always use this for multiplication" nudges the model to use the tool rather than answering directly. Without that instruction, Claude might answer simple questions directly and only use the tool for harder ones.

---

## Experiment Design

### Tests

| Test | Number range | Samples | API calls |
|---|---|---|---|
| 2-digit | 10 – 99 | 100 | ~300 (100 direct + 200 tool) |
| 5-digit | 10,000 – 99,999 | 100 | ~300 |

### Per sample, the script:
1. Generates two random numbers of the target size
2. Computes Python ground truth: `correct = num1 * num2`
3. Calls `ask_llm_direct()` — single API call, no tools
4. Calls `ask_llm_with_tool()` — two-turn conversation with calculator tool
5. Records both results vs. ground truth
6. Writes one row to CSV

### Output columns (CSV)

| Column | Description |
|---|---|
| `sample` | Sample index (1–100) |
| `num1` | First randomly generated number |
| `num2` | Second randomly generated number |
| `python_answer` | Exact result from Python |
| `llm_direct_answer` | What Claude returned without tools |
| `llm_direct_correct` | True/False |
| `llm_tool_answer` | What Claude returned after using the tool |
| `llm_tool_correct` | True/False |
| `tool_was_called` | Whether Claude actually invoked the tool |

The `tool_was_called` column is particularly interesting — occasionally Claude may answer without calling the tool (especially for very simple inputs). This column lets you see how often the agentic pattern was actually triggered.

---

## Architecture

```
run_test_v2.py
    │
    ├── TEST_CONFIGS  ← extensible: add any digit size
    │
    ├── For each sample:
    │   │
    │   ├── Python:              num1 * num2  → always correct
    │   │
    │   ├── ask_llm_direct()
    │   │     └── 1 API call, no tools
    │   │         Claude predicts the answer
    │   │
    │   └── ask_llm_with_tool()
    │         ├── Turn 1: Claude calls calculator(num1, num2)
    │         ├── Python executes: result = num1 * num2
    │         └── Turn 2: Claude receives result, returns answer
    │
    ├── results/results_v2_2digit.csv
    └── results/results_v2_5digit.csv
```

---

## How to Run

```bash
source venv/bin/activate
python3 run_test_v2.py
```

Expected runtime: ~8–10 minutes (400 total API calls with 0.5s delay between samples).

Live output per sample:
```
  [  1]     42        67           2814          2814          2814  ✓  ✓
  [  2]     83        19           1577          1577          1577  ✓  ✓
  [  3]     61        94           5734          5535          5734  ✗  ✓
```
Columns: index, num1, num2, python, direct, tool, direct_ok, tool_ok

---

## Expected Results

| Test | Python | LLM Direct | LLM + Tool |
|---|---|---|---|
| 2-digit | 100% | ~85–95% | ~98–100% |
| 5-digit | 100% | ~5–15% | ~95–100% |

The key finding should be that **LLM + Tool nearly eliminates the accuracy gap** between 2-digit and 5-digit multiplication — because the LLM is no longer doing the arithmetic at all.

---

## Actual Results

*(Run `python3 run_test_v2.py` and fill in below)*

| Test | Range | Samples | Python | LLM Direct | LLM + Tool | Tool Call Rate |
|---|---|---|---|---|---|---|
| 2-digit | 10–99 | 100 | 100% | —% | —% | —% |
| 5-digit | 10,000–99,999 | 100 | 100% | —% | —% | —% |

---

## Key Takeaways

**1. Tool use is the correct fix for arithmetic — not better prompting.**
Chain-of-thought, "show your work", "be careful" — none of these reliably fix arithmetic errors in LLMs. The root cause is architectural: the model predicts tokens, it doesn't compute. The only reliable fix is to remove arithmetic from what the model does.

**2. The tool call rate reveals trust calibration.**
If Claude sometimes answers simple 2-digit multiplications without calling the tool, that's the model being overconfident on easy cases. If it always calls the tool regardless of complexity, that's better-calibrated behavior. The `tool_was_called` column lets you measure this.

**3. The LLM's role in tool use is routing, not computing.**
In the agentic pattern, Claude's job is to understand the user's intent, choose the right tool, and format the inputs correctly. Python's job is to compute. This separation of concerns is the foundation of reliable AI systems.

**4. Two API calls per sample is a real cost consideration.**
Tool use doubles the number of API calls. For production systems at scale, this cost compounds. The tradeoff is: slightly more expensive, dramatically more accurate.

---

## Comparison: v1 vs. v2

| | run_test.py (v1) | run_test_v2.py (v2) |
|---|---|---|
| Approaches | Python + LLM Direct | Python + LLM Direct + LLM + Tool |
| API calls per sample | 1 | 3 (1 direct + 2 tool) |
| Key question | How accurate is the LLM? | Does tool use fix it? |
| New concept | Semantic arithmetic limits | Agentic tool use pattern |

---

## Project Structure

```
LLM Accuracy on Multiplication/
├── run_test.py         ← v1: Python vs LLM Direct
├── run_test_v2.py      ← v2: Python vs LLM Direct vs LLM + Tool
├── requirements.txt
├── .env                ← AWS credentials (git-ignored)
├── .gitignore
├── readme.md           ← v1 documentation
├── readmev2.md         ← this file
└── results/
    ├── results_2digit.csv        ← v1 results
    ├── results_5digit.csv        ← v1 results
    ├── results_v2_2digit.csv     ← v2 results
    └── results_v2_5digit.csv     ← v2 results
```
