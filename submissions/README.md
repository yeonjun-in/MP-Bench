# Submission Guide

## How to Submit

1. Fork this repository
2. Create a folder under `submissions/<your_model_name>/`
3. Add the required files (see below)
4. Open a Pull Request — evaluation runs automatically via GitHub Actions

---

## Required Files

```
submissions/<model_name>/
├── metadata.json          # model info
├── predictions_manual.json     # predictions on the manual split (169 files)
└── predictions_automatic.json  # predictions on the automatic split (126 files)
```

---

## `metadata.json`

```json
{
  "model_name": "GPT-4o",
  "organization": "OpenAI",
  "paper_url": "https://arxiv.org/abs/...",
  "code_url": "https://github.com/...",
  "description": "Optional short description of the system"
}
```

---

## Prediction File Format

Each prediction file is a **JSON object** mapping `file_id` (the filename without `.json`) to a list of **step indices ranked by predicted failure likelihood** (most likely failure first).

```json
{
  "trace_001": [3, 1, 5, 2, 0, 4],
  "trace_002": [0, 2, 1],
  "trace_003": [4, 3, 1, 0, 2, 5, 6]
}
```

- `file_id` must exactly match the filename (without `.json`) of the corresponding annotated file
- Step indices are **0-based** and must reference valid steps in that trace
- Only the **top-5** ranked steps are used for nDCG@5 evaluation; you may include more for safety
- Every file in the split must be present — missing files receive a score of 0

---

## Evaluation Metric

Submissions are ranked by **nDCG@5 (Exp)** computed separately for `manual` and `automatic` splits.

**Exponential gain nDCG@5:**

$$\text{nDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}}, \quad \text{DCG@5} = \sum_{i=1}^{5} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

where `rel_i = 1` if the step at rank `i` is annotated as a failure step, `0` otherwise.

---

## Example Submission

See `submissions/example_gpt4o/` for a minimal working example.
