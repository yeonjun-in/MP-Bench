import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from masevaluator import MASEval
import dotenv
dotenv.load_dotenv()


class GPTAnnotatorJudge:
    """
    LLM-as-a-Judge system for comparing GPT predictions with annotator annotations.
    Evaluates how well GPT's failure attribution matches annotator's reasoning.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 model_type: str = "openai", model_name: str = None, 
                 hf_token: str = None, device: str = "auto"):
        """
        Initialize GPTAnnotatorJudge with model configuration.
        
        Args:
            api_key (str): API key (for openai/together model_type)
            base_url (str, optional): Custom OpenAI API base URL
            model_type (str): "openai", "together", "llama", or "qwen"
            model_name (str, optional): Specific model name to use
            hf_token (str, optional): Hugging Face token for private models
            device (str): Device for local models ("auto", "cpu", "cuda")
        """
        self.evaluator = MASEval(
            model_type=model_type,
            model_name=model_name,
        )
        # Set temperature to 0 for deterministic judgments
        self.evaluator.temperature = 0.0
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_original_data_file(self, annotator_file_path: str) -> Optional[str]:
        """
        Try to find the original data file with history from annotator file path.
        Tries common patterns like annotated/1/mast/1.json, annotated/2/mast/1.json, etc.
        
        Args:
            annotator_file_path: Path to unified annotator file
            
        Returns:
            Path to original data file if found, None otherwise
        """
        path = Path(annotator_file_path)
        
        # Try to find original file in annotated/1/, annotated/2/, annotated/3/
        for annotator_num in ['1', '2', '3']:
            # Replace 'unified' with annotator number
            original_path = path.parent.parent / annotator_num / path.parent.name / path.name
            if original_path.exists():
                return str(original_path)
        
        return None
    
    def build_execution_log(self, data: Dict[str, Any], target_step: int, context_window: int = 5) -> str:
        """
        Build execution log context around the target step.
        
        Args:
            data: JSON data containing history array
            target_step: Step number to focus on
            context_window: Number of steps before and after to include
            
        Returns:
            Formatted execution log string
        """
        history = data.get('history', [])
        if not history:
            return "No execution log available."
        
        context_parts = []
        target_step_int = int(target_step)
        start_step = max(0, target_step_int - context_window)
        end_step = target_step_int + context_window
        
        for step in history:
            step_num = int(step.get('step', '0'))
            if start_step <= step_num <= end_step:
                role = step.get('role', 'unknown')
                content = step.get('content', '')
                context_parts.append(f"[Step {step_num}] Role: {role}\nContent: {content}\n")
        
        if not context_parts:
            return f"No execution log available for step {target_step}."
        
        return "\n".join(context_parts)
    
    def compare_annotations(self, gpt_prediction: Dict[str, Any], 
                           annotator_annotation: Dict[str, Any],
                           question: str, step_num: str,
                           execution_log: str = "") -> Dict[str, Any]:
        """
        Use LLM to compare GPT prediction with annotator annotation for a specific step.
        
        Args:
            gpt_prediction: GPT's prediction dict with 'failure_reason' and 'ideal_action'
            annotator_annotation: Annotator's annotation dict with 'fail_reason' and 'ideal_action'
            question: Original user question
            step_num: Step number being compared
            execution_log: Execution log context around the target step
            
        Returns:
            Dictionary with comparison results
        """
        gpt_fail_reason = gpt_prediction.get('failure_reason', '')
        gpt_ideal_action = gpt_prediction.get('ideal_action', '')
        
        annotator_fail_reason = annotator_annotation.get('fail_reason', '')
        annotator_ideal_action = annotator_annotation.get('ideal_action', '')
        
        prompt = f"""
You are an expert evaluator assessing the quality of a large language model's failure attribution reasoning by comparing it with a human expert annotator's annotation.

Your goal is NOT to check for exact wording matches, but to evaluate whether the model provides a plausible, well-grounded reasoning that aligns with the human annotator's judgment.

---

**Original Task:**
{question}

**Failure Step:**
Step {step_num}

**Execution Log (context around step {step_num}):**
{execution_log}

---

**Model Prediction:**
- Failure Reason: {gpt_fail_reason}
- Ideal Action: {gpt_ideal_action}

**Human Expert Annotation:**
- Failure Reason: {annotator_fail_reason}
- Ideal Action: {annotator_ideal_action}

---

**Evaluation Criteria:**

Evaluate the model's prediction by jointly considering the failure reason and ideal action, based on the following aspects:

1. **Reasoning Alignment**  
   Does the model capture the core reasoning behind why the step is considered a failure and what should have been done instead, as reflected in the human annotation?

2. **Faithfulness to Execution Context**  
   Is the model's reasoning grounded in the actual execution context of this step?
   - Does it rely only on information available in the execution trace?
   - Does it avoid introducing unsupported assumptions or hindsight knowledge?

3. **Coverage and Completeness**  
   Does the model address the key issues and considerations raised by the annotator, even if expressed differently?

4. **Plausibility of Ideal Action**  
   Is the proposed ideal action reasonable, actionable, and consistent with the annotator’s intent?

---

**Scoring Guidelines (1–10 scale):**

- **9–10 (Excellent):**  
  Strong alignment with the annotator’s reasoning; well-grounded, faithful to the execution context, and captures the key rationale and ideal action clearly.

- **7–8 (Good):**  
  Largely aligned with minor omissions or differences; reasoning is plausible and context-faithful.

- **5–6 (Moderate):**  
  Partially aligned but missing important aspects or containing some questionable assumptions.

- **3–4 (Poor):**  
  Weak alignment; reasoning diverges significantly from the annotator or lacks grounding in the execution context.

- **1–2 (Very Poor):**  
  Largely incorrect, unfaithful to the execution context, or fails to provide a meaningful explanation.

---

**Output Format (JSON only):**
{{
    "overall_score": "integer between 1 and 10",
    "fail_reason_score": "integer between 1 and 10",
    "ideal_action_score": "integer between 1 and 10",
    "reasoning": "Concise explanation of the judgment",
    "key_agreements": ["Key points where the model aligns with the annotator"],
    "key_mismatches": ["Key points where the model diverges from the annotator"]
}}

Only return the JSON object. Do not include any additional text.
"""


        messages = [
            {"role": "system", "content": "You are an expert evaluator for comparing failure attributions in multi-agent systems. Provide clear, objective judgments in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response_obj = self.evaluator._chat_completion(messages)
            
            # Extract text from response
            if hasattr(response_obj, 'choices') and len(response_obj.choices) > 0:
                response = response_obj.choices[0].message.content
            elif isinstance(response_obj, str):
                response = response_obj
            else:
                response = str(response_obj)
            
            # Try to parse JSON from response
            # Sometimes LLM wraps JSON in markdown code blocks or includes reasoning tags
            if '</think>' in response:
                response = response.split('</think>')[-1].strip()
            if '</think>' in response:
                response = response.split('</think>')[-1].strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            judgment = json.loads(response)
            
            return {
                "step": step_num,
                "overall_score": judgment.get("overall_score", 0),
                "fail_reason_score": judgment.get("fail_reason_score", 0),
                "ideal_action_score": judgment.get("ideal_action_score", 0),
                "faithfulness_score": judgment.get("faithfulness_score", 0),
                "reasoning": judgment.get("reasoning", ""),
                "key_agreements": judgment.get("key_agreements", []),
                "key_mismatches": judgment.get("key_mismatches", []),
                "gpt_fail_reason": gpt_fail_reason,
                "gpt_ideal_action": gpt_ideal_action,
                "annotator_fail_reason": annotator_fail_reason,
                "annotator_ideal_action": annotator_ideal_action
            }
        except json.JSONDecodeError as e:
            return {
                "step": step_num,
                "overall_score": None,
                "fail_reason_score": None,
                "ideal_action_score": None,
                "faithfulness_score": None,
                "reasoning": f"Failed to parse LLM response as JSON: {str(e)}",
                "key_agreements": [],
                "key_mismatches": [],
                "gpt_fail_reason": gpt_fail_reason,
                "gpt_ideal_action": gpt_ideal_action,
                "annotator_fail_reason": annotator_fail_reason,
                "annotator_ideal_action": annotator_ideal_action,
                "raw_response": response
            }
        except Exception as e:
            return {
                "step": step_num,
                "overall_score": None,
                "fail_reason_score": None,
                "ideal_action_score": None,
                "faithfulness_score": None,
                "reasoning": f"Error during LLM evaluation: {str(e)}",
                "key_agreements": [],
                "key_mismatches": [],
                "gpt_fail_reason": gpt_fail_reason,
                "gpt_ideal_action": gpt_ideal_action,
                "annotator_fail_reason": annotator_fail_reason,
                "annotator_ideal_action": annotator_ideal_action
            }
    
    def evaluate_files(self, gpt_file_path: str, annotator_file_path: str) -> Dict[str, Any]:
        """
        Compare GPT predictions with annotator annotations across all common steps.
        
        Args:
            gpt_file_path: Path to GPT prediction JSON file
            annotator_file_path: Path to annotator annotation JSON file
            
        Returns:
            Dictionary with comparison results
        """
        gpt_data = self.load_json_file(gpt_file_path)
        annotator_data = self.load_json_file(annotator_file_path)
        
        question = gpt_data.get('question', '') or annotator_data.get('question', '')
        
        gpt_predictions = gpt_data.get('unified_failure_attribution', {})
        annotator_annotations = annotator_data.get('final_fail_annotations', {})
        
        # Find common steps
        gpt_steps = set(gpt_predictions.keys())
        annotator_steps = set(annotator_annotations.keys())
        common_steps = gpt_steps.intersection(annotator_steps)
        gpt_only_steps = gpt_steps - annotator_steps
        annotator_only_steps = annotator_steps - gpt_steps
        
        results = {
            "gpt_file_path": gpt_file_path,
            "annotator_file_path": annotator_file_path,
            "question": question,
            "total_gpt_steps": len(gpt_steps),
            "total_annotator_steps": len(annotator_steps),
            "common_steps": len(common_steps),
            "gpt_only_steps": list(gpt_only_steps),
            "annotator_only_steps": list(annotator_only_steps),
            "comparisons": []
        }
        
        # Try to find original data file with history
        original_data_file = os.path.join(annotator_file_path.split('/')[0], '1', '/'.join(annotator_file_path.split('/')[2:]))
        original_data = None
        if original_data_file:
            try:
                original_data = self.load_json_file(original_data_file)
            except Exception as e:
                print(f"Warning: Could not load original data file {original_data_file}: {e}")
        
        # Compare common steps
        for step_num in sorted(common_steps, key=lambda x: int(x)):
            gpt_pred = gpt_predictions[step_num]
            annotator_ann = annotator_annotations[step_num]
            
            # Build execution log if original data is available
            execution_log = ""
            if original_data:
                execution_log = self.build_execution_log(original_data, int(step_num))
            else:
                execution_log = "No execution log available."
            
            comparison = self.compare_annotations(
                gpt_pred, annotator_ann, question, step_num, execution_log
            )
            results["comparisons"].append(comparison)
        
        # Calculate summary statistics
        valid_scores = [c["overall_score"] for c in results["comparisons"] 
                       if c["overall_score"] is not None]
        valid_fail_reason_scores = [c["fail_reason_score"] for c in results["comparisons"] 
                                   if c["fail_reason_score"] is not None]
        valid_ideal_action_scores = [c["ideal_action_score"] for c in results["comparisons"] 
                                    if c["ideal_action_score"] is not None]
        valid_faithfulness_scores = [c["faithfulness_score"] for c in results["comparisons"] 
                                    if c["faithfulness_score"] is not None]
        
        alignment_counts = {}
        for c in results["comparisons"]:
            alignment = c.get("overall_alignment", "unknown")
            alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1
        
        results["summary"] = {
            "average_overall_score": sum(valid_scores) / len(valid_scores) if valid_scores else None,
            "average_fail_reason_score": sum(valid_fail_reason_scores) / len(valid_fail_reason_scores) if valid_fail_reason_scores else None,
            "average_ideal_action_score": sum(valid_ideal_action_scores) / len(valid_ideal_action_scores) if valid_ideal_action_scores else None,
            "average_faithfulness_score": sum(valid_faithfulness_scores) / len(valid_faithfulness_scores) if valid_faithfulness_scores else None,
            "alignment_distribution": alignment_counts,
            "total_comparisons": len(results["comparisons"]),
            "valid_comparisons": len(valid_scores),
            "error_comparisons": len(results["comparisons"]) - len(valid_scores)
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for comparing GPT predictions with annotator annotations")
    parser.add_argument("--gpt_file", type=str, 
                       default="results/openai_gpt_4.1/all_at_once_taxonomy/unified_claude-sonnet-4-5/mast/1.json",
                       help="Path to GPT prediction JSON file")
    parser.add_argument("--annotator_file", type=str,
                       default="annotated/unified_claude-sonnet-4-5/mast/1.json",
                       help="Path to annotator annotation JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: gpt_file_comparison.json)")
    parser.add_argument("--model_type", type=str, default="claude",
                       choices=["openai", "together", "llama", "qwen", "claude"])
    parser.add_argument("--model_name", type=str, default='claude-sonnet-4-5',
                       help="Model name for LLM-as-a-Judge")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"])
    
    args = parser.parse_args()
    if 'claude' in args.model_name:
        args.model_type = 'claude'
    else:
        args.model_type = 'openai'
    # Initialize judge
    judge = GPTAnnotatorJudge(
        model_type=args.model_type,
        model_name=args.model_name,
    )
    
    # Evaluate files
    print(f"Comparing GPT predictions with annotator annotations...")
    print(f"GPT file: {args.gpt_file}")
    print(f"Annotator file: {args.annotator_file}")
    
    results = judge.evaluate_files(args.gpt_file, args.annotator_file)
    
    # Save results
    if args.output is None:
        # Save to eval_results_$JUDGE_MODEL folder
        gpt_file_path = Path(args.gpt_file)
        # Replace 'results' with 'eval_results_$JUDGE_MODEL'
        path_parts = list(gpt_file_path.parts)
        if path_parts[0] == 'results':
            path_parts[0] = f'eval_results_{args.model_name}'
        else:
            # If path doesn't start with 'results', prepend eval_results_$JUDGE_MODEL
            path_parts = [f'eval_results_{args.model_name}'] + path_parts
        output_path = Path(*path_parts[:-1]) / f'{gpt_file_path.name}'
        output_path = str(output_path)
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    judge.save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total GPT steps: {results['total_gpt_steps']}")
    print(f"  Total Annotator steps: {results['total_annotator_steps']}")
    print(f"  Common steps compared: {results['common_steps']}")
    print(f"  GPT-only steps: {len(results['gpt_only_steps'])}")
    print(f"  Annotator-only steps: {len(results['annotator_only_steps'])}")
    
    if results['summary']['average_overall_score'] is not None:
        print(f"\n  Average Overall Score: {results['summary']['average_overall_score']:.2f}")
        print(f"  Average Fail Reason Score: {results['summary']['average_fail_reason_score']:.2f}")
        print(f"  Average Ideal Action Score: {results['summary']['average_ideal_action_score']:.2f}")
        print(f"  Average Faithfulness Score: {results['summary']['average_faithfulness_score']:.2f}")
        print(f"  Alignment Distribution: {results['summary']['alignment_distribution']}")
        print(f"  Valid comparisons: {results['summary']['valid_comparisons']}")
        print(f"  Error comparisons: {results['summary']['error_comparisons']}")

