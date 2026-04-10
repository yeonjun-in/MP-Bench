# MP-Bench: Multi-Perspective Failure Attribution Benchmark for Multi-Agent Systems

MP-Bench is a benchmark and evaluation pipeline for multi-perspective failure attribution in multi-agent systems. This repository focuses on two goals:

1. reconstructing raw execution traces plus human step-level annotations into a unified benchmark format
2. evaluating LLM-based failure attribution systems against consolidated human annotations

For broader benchmark context and task provenance, see the upstream benchmark reference at [adobe-research/multi-agent-eval-bench](https://github.com/adobe-research/multi-agent-eval-bench).


## Dataset Overview
The raw benchmark entries are expected under `MP-Bench/<annotator_id>/<split>/<file>.json`. Note that this directory is empty in the repository; you need to download the raw annotated dataset from the following [link](https://github.com/adobe-research/multi-agent-eval-bench/tree/main/MP-Bench) and place the files accordingly.


Each raw file contains:
- `log_source`: pointer to the original conversation trace
- `annotation`: step-level human annotations for failure attribution

The `log_source` can point to different source types:
- JSON conversation logs
- plain-text `console_log.txt` traces

`build_annotated_from_log_source.py` normalizes both formats into a unified JSON structure with:
- `question`
- `ground_truth`
- `history`
- step-level annotation fields such as `fail_annotation`, `fail_category`, `fail_reason`, and `ideal_action`

In the current evaluation scripts, the benchmark is split into:
- `manual`: 169 files
- `automatic`: 126 files


## Normalized Data Format
After preprocessing, each example is stored as a JSON file with the following high-level structure:

```json
{
  "is_correct": null,
  "question": "...",
  "ground_truth": "...",
  "history": [
    {
      "content": "...",
      "role": "...",
      "step": "0",
      "fail_annotation": "0 or 1",
      "fail_category": "...",
      "fail_reason": "...",
      "ideal_action": "..."
    }
  ],
  "system_prompt": {}
}
```

This normalized format is what the rest of the pipeline expects when reconstructing and auditing traces.

## Repository Layout
```text
.
├── MP-Bench/                         # raw benchmark metadata per annotator
├── build_annotated_from_log_source.py
├── data_processing.sh
├── masevaluator.py
├── run_maseval.py
├── run.sh
├── reasoning_consolidation.py
├── evaluate_gpt_vs_annotator.py
└── run_gpt_annotator_comparison.sh
```

## Environment Setup
Create the conda environment:

```bash
conda env create -f environment.yml
conda activate mpbench
```

Set API keys in `.env` as needed:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
TOGETHER_API_KEY=...
HF_TOKEN=...
```


## Step 1: Build The Benchmark Files
First, download the raw benchmark data and place the raw `MP-Bench/` folder at the repository root.

Then run:

```bash
bash data_processing.sh
```

This script:
- scans every file under `MP-Bench/*/automatic/*.json` and `MP-Bench/*/manual/*.json`
- resolves each `log_source`
- reconstructs the full execution trace
- saves normalized files to `annotated/<annotator_id>/<split>/<file>.json`

This is the stage that turns pointer-based metadata into a usable benchmark with full logs and aligned step annotations.

## Step 2: Evaluation Protocol
The evaluation pipeline in this repository has four stages.

### 1. Failure Attribution Generation
`run.sh` runs the failure attribution model over an entire benchmark split.

Command template:

```bash
sh run.sh <model_type> <model_name> <method> <seed> <split> <gpu_id>
```

Example:

```bash
sh run.sh openai gpt-5.1 all_at_once_taxonomy 0 manual 0
sh run.sh openai gpt-5.1 all_at_once_taxonomy 1 manual 0
sh run.sh openai gpt-5.1 all_at_once_taxonomy 2 manual 0
```

Important notes:
- the current script evaluates `manual` and `automatic` separately
- the protocol uses multiple random seeds (typically `0`, `1`, `2`)
- outputs are written to `results/<model>/<method>/seed_<seed>/<split>/`

### 2. Consolidation
`reasoning_consolidation.py` performs two kinds of consolidation.

Human-side consolidation:
- reads per-annotator benchmark files from `annotated/1`, `annotated/2`, and `annotated/3`
- merges multiple annotator judgments for the same step
- uses an LLM to summarize disagreements when multiple annotators flagged the same step
- writes consolidated references to `annotated/unified_<model_name>/<split>/`

Model-side consolidation:
- reads model predictions across multiple seeds from `results/<backbone>/<method>/seed_<seed>/...`
- merges repeated failure attributions across seeds
- uses an LLM to summarize multi-seed reasoning
- writes consolidated model outputs to `results/<backbone>/<method>/unified_<model_name>/<split>/`

Example:

```bash
python reasoning_consolidation.py --model_type openai --model_name gpt-5.1
```

### 3. Failure Attribution Evaluation (Section 5.1)

To evaluate the failure attribution capabilities of LLM-based systems, run `failure_attribution_evaluation_display.ipynb`.

### 4. Attribution Reasoning Evaluation (Section 5.2)
Finally, the repository evaluates how well the model's failure attribution reasoning matches the consolidated human annotations using an LLM-as-a-Judge protocol.

Batch command template:

```bash
sh run_gpt_annotator_comparison.sh <model_folder> <split> <judge_model> <method> <consolidation_model>
```

Example:

```bash
sh run_gpt_annotator_comparison.sh openai_gpt_5.1 manual gpt-5.1 all_at_once_taxonomy gpt-5.1
sh run_gpt_annotator_comparison.sh openai_gpt_5.1 automatic gpt-5.1 all_at_once_taxonomy gpt-5.1
```

This stage:
- aligns common failure steps between model predictions and human annotations
- provides the execution context around each target step
- scores agreement between model reasoning and annotator reasoning

The judge outputs include:
- `overall_score`
- `fail_reason_score`
- `ideal_action_score`
- textual reasoning
- agreement and mismatch summaries

Outputs are written under:

```text
eval_results_<judge_model>/<model_folder>/<method>/unified_<consolidation_model>/<split>/
```

To view the attribution reasoning scores for each LLM-based system, run `failure_attribution_evaluation_display.ipynb`.

## Recommended End-to-End Workflow
```bash
# 1) download data from https://github.com/adobe-research/multi-agent-eval-bench/MP-Bench

# 2) reconstruct normalized benchmark files
bash data_processing.sh

# 3) run failure attribution for multiple seeds
sh run.sh openai gpt-5.1 all_at_once_taxonomy 0 manual 0
sh run.sh openai gpt-5.1 all_at_once_taxonomy 1 manual 0
sh run.sh openai gpt-5.1 all_at_once_taxonomy 2 manual 0

# 4) consolidate human annotations and model predictions
python reasoning_consolidation.py --model_type openai --model_name gpt-5.1

# 5) evaluate model reasoning against consolidated annotations
sh run_gpt_annotator_comparison.sh openai_gpt_5.1 manual gpt-5.1 all_at_once_taxonomy gpt-5.1
sh run_gpt_annotator_comparison.sh openai_gpt_5.1 automatic gpt-5.1 all_at_once_taxonomy gpt-5.1

# 6) to view each metric value, run failure_attribution_evaluation_display.ipynb and attribution_reasoning_evaluation_display.ipynb.
```


## Output Summary
- `annotated/<annotator_id>/<split>/`: normalized per-annotator benchmark files
- `annotated/unified_<model_name>/<split>/`: consolidated human references
- `results/<model>/<method>/seed_<seed>/<split>/`: raw model predictions
- `results/<model>/<method>/unified_<model_name>/<split>/`: consolidated model predictions
- `eval_results_<judge_model>/...`: LLM-as-a-Judge comparison results

## 📝 Citation

```bibtex
@article{in2026rethinking,
  title={Rethinking Failure Attribution in Multi-Agent Systems: A Multi-Perspective Benchmark and Evaluation},
  author={In, Yeonjun and Tanjim, Mehrab and Subramanian, Jayakumar and Kim, Sungchul and Bhattacharya, Uttaran and Kim, Wonjoong and Park, Sangwu and Sarkhel, Somdeb and Park, Chanyoung},
  journal={arXiv preprint arXiv:2603.25001},
  year={2026}
}
```