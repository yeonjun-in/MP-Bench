# %%


# %%
import json, os
from collections import defaultdict
from masevaluator import MASEval
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Reasoning Consolidation")
parser.add_argument("--model_type", type=str, default="openai",
                    choices=["openai", "together", "llama", "qwen", 'claude'])
parser.add_argument("--model_name", type=str, default="gpt-5.1",
                    help="Model name for reasoning consolidation")
args = parser.parse_args()

def process_annotations(task_type, file_id):
    """
    주어진 task_type과 file_id에 대해 3개 annotator의 annotation을 통합 처리
    
    Args:
        task_type: 'automatic', 'manual'
        file_id: 파일 ID (예: '1', '2', ...)
    
    Returns:
        dict: 처리 결과 또는 None (파일이 없을 경우)
    """
    # 파일 경로
    file1 = f'annotated/1/{task_type}/{file_id}.json'
    file2 = f'annotated/2/{task_type}/{file_id}.json'
    file3 = f'annotated/3/{task_type}/{file_id}.json'
    
    # 파일 존재 확인
    if not all(os.path.exists(f) for f in [file1, file2, file3]):
        return None
    
    # 데이터 로드
    try:
        data1 = json.load(open(file1))
        data2 = json.load(open(file2))
        data3 = json.load(open(file3))
    except Exception as e:
        print(f"Error loading {task_type}/{file_id}.json: {e}")
        return None
    
    # 각 step별로 fail로 표시한 annotator 수를 계산
    step_fail_count = defaultdict(list)
    
    # data1에서 fail step 수집
    for step_data in data1['history']:
        if step_data['fail_annotation'] == '1':
            step = step_data['step']
            step_fail_count[step].append({
                'annotator': 1,
                'fail_reason': step_data.get('fail_reason', ''),
                'ideal_action': step_data.get('ideal_action', ''),
                'fail_category': step_data.get('fail_category', '')
            })
    
    # data2에서 fail step 수집
    for step_data in data2['history']:
        if step_data['fail_annotation'] == '1':
            step = step_data['step']
            step_fail_count[step].append({
                'annotator': 2,
                'fail_reason': step_data.get('fail_reason', ''),
                'ideal_action': step_data.get('ideal_action', ''),
                'fail_category': step_data.get('fail_category', '')
            })
    
    # data3에서 fail step 수집
    for step_data in data3['history']:
        if step_data['fail_annotation'] == '1':
            step = step_data['step']
            step_fail_count[step].append({
                'annotator': 3,
                'fail_reason': step_data.get('fail_reason', ''),
                'ideal_action': step_data.get('ideal_action', ''),
                'fail_category': step_data.get('fail_category', '')
            })
    
    # 여러 annotator가 동시에 fail로 표시한 step과 유일하게 fail로 표시한 step 분리
    multi_annotator_fails = {step: annotators for step, annotators in step_fail_count.items() if len(annotators) > 1}
    single_annotator_fails = {step: annotators[0] for step, annotators in step_fail_count.items() if len(annotators) == 1}
    
    # 각 step별 annotator 수 정보
    step_annotator_count = {step: len(annotators) for step, annotators in step_fail_count.items()}
    
    return {
        'data1': data1,
        'data2': data2,
        'data3': data3,
        'multi_annotator_fails': multi_annotator_fails,
        'single_annotator_fails': single_annotator_fails,
        'step_annotator_count': step_annotator_count
    }


# %%
def summarize_fails(evaluator, multi_annotator_fails, data1):
    """
    여러 annotator가 동시에 fail로 표시한 step들을 LLM으로 summarize
    
    Args:
        evaluator: MASEval 인스턴스
        multi_annotator_fails: 여러 annotator가 동시에 fail로 표시한 step들
        data1: 첫 번째 annotator의 데이터 (execution log용)
    
    Returns:
        dict: summarized_fails
    """
    summarized_fails = {}
    
    for step, annotators in multi_annotator_fails.items():
        # 각 annotator의 fail_reason과 ideal_action을 수집
        fail_reasons = []
        ideal_actions = []
        fail_categories = []
        
        for ann in annotators:
            if ann['fail_reason']:
                fail_reasons.append(f"Annotator {ann['annotator']}: {ann['fail_reason']}")
            if ann['ideal_action']:
                ideal_actions.append(f"Annotator {ann['annotator']}: {ann['ideal_action']}")
            if ann['fail_category']:
                fail_categories.append(ann['fail_category'])
        
        # 해당 step의 execution log 가져오기 (masevaluator의 format_chat_content 사용)
        # 해당 step까지의 history를 포함하여 컨텍스트 제공
        step_idx = None
        for idx, step_data in enumerate(data1['history']):
            if step_data['step'] == step:
                step_idx = idx
                break
        
        # 해당 step과 그 이전 몇 개 step을 포함 (최대 5개 step, 또는 해당 step까지)
        if step_idx is not None:
            start_idx = max(0, step_idx - 4)  # 최대 5개 step 포함
            context_history = data1['history'][start_idx:step_idx + 1]
            execution_log = evaluator.format_chat_content(context_history)
        else:
            execution_log = "No execution log available"
        
        # LLM에 요약 요청
        prompt = f"""You are summarizing failure annotations from multiple annotators for step {step}.

Execution Log (context leading up to and including step {step}):
{execution_log}

Fail Reasons from different annotators:
{chr(10).join(fail_reasons) if fail_reasons else 'No fail reasons provided'}

Ideal Actions from different annotators:
{chr(10).join(ideal_actions) if ideal_actions else 'No ideal actions provided'}

Fail Categories: {', '.join(set(fail_categories)) if fail_categories else 'None'}

Please provide a comprehensive summary that incorporates all annotators' perspectives:
1. A unified fail_reason that captures the essence of all annotators' concerns
2. A unified ideal_action that reflects the best practices suggested by all annotators
3. A fail_category (choose the most appropriate one from: {', '.join(set(fail_categories)) if fail_categories else 'Tool/Agent Invocation Errors, Verification Errors, Planning Errors, Communication Errors'} or suggest a new one)

Respond in JSON format:
{{
    "fail_reason": "...",
    "ideal_action": "...",
    "fail_category": "..."
}}"""

        messages = [
            {"role": "system", "content": "You are an expert at synthesizing multiple perspectives into coherent summaries. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = evaluator._chat_completion(messages)
            
            # Extract text from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
            
            # Parse JSON (handle markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            summary = json.loads(response_text)
            summarized_fails[step] = summary
        except Exception as e:
            print(f"    Error summarizing step {step}: {e}")
            # Fallback: use the first annotator's annotation
            summarized_fails[step] = {
                'fail_reason': annotators[0]['fail_reason'],
                'ideal_action': annotators[0]['ideal_action'],
                'fail_category': annotators[0]['fail_category']
            }
    
    return summarized_fails


# %%
def create_final_annotations(summarized_fails, single_annotator_fails, step_annotator_count):
    """
    최종 fail annotations 생성
    
    Args:
        summarized_fails: LLM으로 summarize한 결과
        single_annotator_fails: 단독 annotator가 fail로 표시한 step들
        step_annotator_count: 각 step별 annotator 수 정보
    
    Returns:
        dict: final_fail_annotations
    """
    final_fail_annotations = {}
    
    # 여러 annotator가 동시에 fail로 표시한 step: LLM으로 summarize한 결과 사용
    for step, summary in summarized_fails.items():
        final_fail_annotations[step] = {
            **summary,
            'annotator_count': step_annotator_count.get(step, 0)
        }
    
    # 유일하게 fail로 표시한 step: 해당 annotator의 의견 그대로 사용
    for step, annotation in single_annotator_fails.items():
        final_fail_annotations[step] = {
            'fail_reason': annotation['fail_reason'],
            'ideal_action': annotation['ideal_action'],
            'fail_category': annotation['fail_category'],
            'annotator_count': step_annotator_count.get(step, 1)
        }
    
    return final_fail_annotations


# %%
# 모든 파일에 대해 처리
evaluator = MASEval(model_type=args.model_type, model_name=args.model_name)

task_types = ['automatic', 'manual']

for task_type in task_types:
    print(f"\n{'='*60}")
    print(f"Processing {task_type}...")
    print(f"{'='*60}")
    
    # 해당 task_type의 모든 파일 찾기 (annotated/1에서 기준으로)
    base_dir = Path(f'annotated/1/{task_type}')
    if not base_dir.exists():
        print(f"Skipping {task_type} - directory does not exist")
        continue
    
    # 출력 디렉토리 생성
    output_dir = Path(f'annotated/unified_{args.model_name}/{task_type}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모든 JSON 파일 찾기
    json_files = sorted(base_dir.glob('*.json'), key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))
    
    for json_file in json_files:
        file_id = json_file.stem
        output_path = output_dir / f'{file_id}.json'
        
        # 이미 완료된 파일은 스킵
        if output_path.exists():
            print(f"\nSkipping {task_type}/{file_id}.json (already exists)")
            continue
        
        print(f"\nProcessing {task_type}/{file_id}.json...")
        
        # Annotation 처리
        result = process_annotations(task_type, file_id)
        if result is None:
            print(f"  Skipped - not all annotator files exist")
            continue
        
        data1 = result['data1']
        multi_annotator_fails = result['multi_annotator_fails']
        single_annotator_fails = result['single_annotator_fails']
        step_annotator_count = result['step_annotator_count']
        
        # LLM으로 summarize
        if multi_annotator_fails:
            print(f"  Summarizing {len(multi_annotator_fails)} steps with multiple annotators...")
            summarized_fails = summarize_fails(evaluator, multi_annotator_fails, data1)
        else:
            summarized_fails = {}
        
        # 최종 annotations 생성
        final_fail_annotations = create_final_annotations(summarized_fails, single_annotator_fails, step_annotator_count)
        
        # 결과 저장
        output_data = {
            'question': data1['question'],
            'ground_truth': data1['ground_truth'],
            'final_fail_annotations': final_fail_annotations,
            'summary': {
                'total_fail_steps': len(final_fail_annotations),
                'multi_annotator_steps': len(multi_annotator_fails),
                'single_annotator_steps': len(single_annotator_fails)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved to {output_path} ({len(final_fail_annotations)} fail steps)")

print("\n" + "="*60)
print("All processing complete!")
print("="*60)


# %% [markdown]
# ### Prediction (Results Processing)

# %%
def process_results(backbone, task_type, file_id, max_seeds=5):
    """
    주어진 backbone, task_type, file_id에 대해 여러 seed의 failure_attribution을 통합 처리
    
    Args:
        backbone: backbone model 이름 (예: 'openai_gpt_4.1')
        task_type: 'automatic', 'manual'
        file_id: 파일 ID (예: '1', '2', ...)
        max_seeds: 최대 seed 수 (기본값: 5, seed_0 ~ seed_4)
    
    Returns:
        dict: 처리 결과 또는 None (파일이 없을 경우)
    """
    # 각 seed의 파일 경로
    if task_type == 'automatic':
        subfolder = 'automatic'
    else:
        subfolder = 'manual'
    
    seed_files = []
    seed_data = {}
    
    for seed in range(max_seeds):
        file_path = f'results/{backbone}/all_at_once_taxonomy/seed_{seed}/{subfolder}/{file_id}.json'
        if os.path.exists(file_path):
            try:
                data = json.load(open(file_path))
                seed_files.append(file_path)
                seed_data[seed] = data
            except Exception as e:
                print(f"    Error loading seed_{seed} file: {e}")
                continue
    
    # 최소 1개 seed 파일이 있어야 함
    if len(seed_files) == 0:
        return None
    
    # 각 step별로 fail로 표시한 seed 수를 계산
    step_fail_count = defaultdict(list)
    
    # 각 seed에서 failure_attribution 수집
    for seed, data in seed_data.items():
        if 'failure_attribution' not in data:
            continue
        
        for failure in data['failure_attribution']:
            step = str(failure.get('step_number', ''))
            if step:
                step_fail_count[step].append({
                    'seed': seed,
                    'fail_reason': failure.get('failure_reason', ''),
                    'ideal_action': failure.get('ideal_action', ''),
                    'agent_name': failure.get('agent_name', '')
                })
    
    # 여러 seed가 동시에 fail로 표시한 step과 유일하게 fail로 표시한 step 분리
    multi_seed_fails = {step: seeds for step, seeds in step_fail_count.items() if len(seeds) > 1}
    single_seed_fails = {step: seeds[0] for step, seeds in step_fail_count.items() if len(seeds) == 1}
    
    # 각 step별 seed 수 정보
    step_seed_count = {step: len(seeds) for step, seeds in step_fail_count.items()}
    
    # 첫 번째 seed의 데이터를 execution log용으로 사용
    first_seed_data = seed_data[min(seed_data.keys())] if seed_data else None
    
    return {
        'seed_data': seed_data,
        'first_seed_data': first_seed_data,
        'multi_seed_fails': multi_seed_fails,
        'single_seed_fails': single_seed_fails,
        'step_seed_count': step_seed_count
    }


# %%
def summarize_fails_from_results(evaluator, multi_seed_fails, first_seed_data):
    """
    여러 seed가 동시에 fail로 표시한 step들을 LLM으로 summarize
    
    Args:
        evaluator: MASEval 인스턴스
        multi_seed_fails: 여러 seed가 동시에 fail로 표시한 step들
        first_seed_data: 첫 번째 seed의 데이터 (execution log용)
    
    Returns:
        dict: summarized_fails
    """
    summarized_fails = {}
    
    for step, seeds in multi_seed_fails.items():
        # 각 seed의 fail_reason과 ideal_action을 수집
        fail_reasons = []
        ideal_actions = []
        agent_names = []
        
        for seed_info in seeds:
            if seed_info['fail_reason']:
                fail_reasons.append(f"Seed {seed_info['seed']}: {seed_info['fail_reason']}")
            if seed_info['ideal_action']:
                ideal_actions.append(f"Seed {seed_info['seed']}: {seed_info['ideal_action']}")
            if seed_info['agent_name']:
                agent_names.append(seed_info['agent_name'])
        
        # 해당 step의 execution log 가져오기 (masevaluator의 format_chat_content 사용)
        step_idx = None
        if first_seed_data and 'history' in first_seed_data:
            for idx, step_data in enumerate(first_seed_data['history']):
                if str(step_data.get('step', '')) == step:
                    step_idx = idx
                    break
            
            # 해당 step과 그 이전 몇 개 step을 포함 (최대 5개 step, 또는 해당 step까지)
            if step_idx is not None:
                start_idx = max(0, step_idx - 4)  # 최대 5개 step 포함
                context_history = first_seed_data['history'][start_idx:step_idx + 1]
                execution_log = evaluator.format_chat_content(context_history)
            else:
                execution_log = "No execution log available"
        else:
            execution_log = "No execution log available"
        
        # LLM에 요약 요청
        prompt = f"""You are summarizing failure attributions from multiple seeds (runs) for step {step}.

Execution Log (context leading up to and including step {step}):
{execution_log}

Fail Reasons from different seeds:
{chr(10).join(fail_reasons) if fail_reasons else 'No fail reasons provided'}

Ideal Actions from different seeds:
{chr(10).join(ideal_actions) if ideal_actions else 'No ideal actions provided'}

Agent Names: {', '.join(set(agent_names)) if agent_names else 'None'}

Please provide a comprehensive summary that incorporates all seeds' perspectives:
1. A unified failure_reason that captures the essence of all seeds' concerns
2. A unified ideal_action that reflects the best practices suggested by all seeds

Respond in JSON format:
{{
    "failure_reason": "...",
    "ideal_action": "..."
}}"""

        messages = [
            {"role": "system", "content": "You are an expert at synthesizing multiple perspectives into coherent summaries. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = evaluator._chat_completion(messages)
            
            # Extract text from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
            
            # Parse JSON (handle markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            summary = json.loads(response_text)
            summarized_fails[step] = summary
        except Exception as e:
            print(f"    Error summarizing step {step}: {e}")
            # Fallback: use the first seed's annotation
            summarized_fails[step] = {
                'failure_reason': seeds[0]['fail_reason'],
                'ideal_action': seeds[0]['ideal_action']
            }
    
    return summarized_fails


# %%
def create_final_annotations_from_results(summarized_fails, single_seed_fails, step_seed_count):
    """
    최종 fail annotations 생성 (results용)
    
    Args:
        summarized_fails: LLM으로 summarize한 결과
        single_seed_fails: 단독 seed가 fail로 표시한 step들
        step_seed_count: 각 step별 seed 수 정보
    
    Returns:
        dict: final_fail_annotations
    """
    final_fail_annotations = {}
    
    # 여러 seed가 동시에 fail로 표시한 step: LLM으로 summarize한 결과 사용
    for step, summary in summarized_fails.items():
        final_fail_annotations[step] = {
            'failure_reason': summary.get('failure_reason', ''),
            'ideal_action': summary.get('ideal_action', ''),
            'seed_count': step_seed_count.get(step, 0)
        }
    
    # 유일하게 fail로 표시한 step: 해당 seed의 의견 그대로 사용
    for step, seed_info in single_seed_fails.items():
        final_fail_annotations[step] = {
            'failure_reason': seed_info['fail_reason'],
            'ideal_action': seed_info['ideal_action'],
            'seed_count': step_seed_count.get(step, 1)
        }
    
    return final_fail_annotations


# %%
# Results 폴더에 대해 처리
evaluator = MASEval(model_type=args.model_type, model_name=args.model_name)

# Configuration
backbones = ['openai_gpt_4.1', 'openai_o3_mini', 'qwen_Qwen3_8B', 'together_openai_gpt_oss_120b', 'openai_gpt_5.1', 'claude_claude_sonnet_4_5']
task_types = ['automatic', 'manual']
max_seeds = 3  # seed_0 ~ seed_4

for backbone in backbones:
    for task_type in task_types:
        print(f"\n{'='*60}")
        print(f"Processing {backbone} - {task_type}...")
        print(f"{'='*60}")
        
        # 해당 backbone/task의 모든 파일 찾기 (seed_0에서 기준으로)
        if task_type == 'automatic':
            subfolder = 'automatic'
        else:
            subfolder = 'manual'
        
        base_dir = Path(f'results/{backbone}/all_at_once_taxonomy/seed_0/{subfolder}')
        if not base_dir.exists():
            print(f"Skipping {backbone}/{task_type} - directory does not exist")
            continue
        
        # 출력 디렉토리 생성
        output_dir = Path(f'results/{backbone}/all_at_once_taxonomy/unified_{args.model_name}/{task_type}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모든 JSON 파일 찾기
        json_files = sorted(base_dir.glob('*.json'), key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))
        
        for json_file in json_files:
            file_id = json_file.stem
            output_path = output_dir / f'{file_id}.json'
            
            # 이미 완료된 파일은 스킵
            if output_path.exists():
                print(f"\nSkipping {backbone}/{task_type}/{file_id}.json (already exists)")
                continue
            
            print(f"\nProcessing {backbone}/{task_type}/{file_id}.json...")
            
            # Results 처리
            result = process_results(backbone, task_type, file_id, max_seeds=max_seeds)
            if result is None:
                print(f"  Skipped - no seed files found")
                continue
            
            first_seed_data = result['first_seed_data']
            multi_seed_fails = result['multi_seed_fails']
            single_seed_fails = result['single_seed_fails']
            step_seed_count = result['step_seed_count']
            
            # LLM으로 summarize
            if multi_seed_fails:
                print(f"  Summarizing {len(multi_seed_fails)} steps with multiple seeds...")
                summarized_fails = summarize_fails_from_results(evaluator, multi_seed_fails, first_seed_data)
            else:
                summarized_fails = {}
            
            # 최종 annotations 생성
            final_fail_annotations = create_final_annotations_from_results(
                summarized_fails, single_seed_fails, step_seed_count
            )
            
            # 결과 저장
            if first_seed_data:
                output_data = {
                    'question': first_seed_data.get('question', ''),
                    'ground_truth': first_seed_data.get('ground_truth', ''),
                    'unified_failure_attribution': final_fail_annotations,
                    'summary': {
                        'total_fail_steps': len(final_fail_annotations),
                        'multi_seed_steps': len(multi_seed_fails),
                        'single_seed_steps': len(single_seed_fails)
                    }
                }
                
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"  ✓ Saved to {output_path} ({len(final_fail_annotations)} fail steps)")

print("\n" + "="*60)
print("All results processing complete!")
print("="*60)


# %% [markdown]
# 


