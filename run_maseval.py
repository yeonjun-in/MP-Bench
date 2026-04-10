from masevaluator import MASEval
import os
import argparse
import random
import numpy as np

def get_model_folder_name(model_type, model_name):
    if model_type == "openai":
        return f"openai_{model_name.replace('/', '_').replace('-', '_')}"
    elif model_type == "together":
        return f"together_{model_name.replace('/', '_').replace('-', '_')}"
    elif model_type == "claude":
        return f"claude_{model_name.replace('/', '_').replace('-', '_')}"
    elif model_type == "llama":
        return f"llama_{model_name.split('/')[-1].replace('-', '_')}"
    elif model_type == "qwen":
        return f"qwen_{model_name.split('/')[-1].replace('-', '_')}"
    else:
        return f"{model_type}_{model_name.replace('/', '_').replace('-', '_')}"

parser = argparse.ArgumentParser(description="Run MAS evaluation with different models")
parser.add_argument("--method", type=str, default="all_at_once_taxonomy", help="Evaluation method")
parser.add_argument("--dataset", type=str, default="manual/1.json", help="Dataset path")
parser.add_argument("--gt", action="store_true", default=False, help="Use ground truth")
parser.add_argument("--seed", type=int, default=0, help="seed number")

# 모델 관련 파라미터
parser.add_argument("--model_type", type=str, default="claude", 
                   choices=["openai", "together", "llama", "qwen", "claude"], 
                   help="Model type: openai, together, llama, qwen, claude")
parser.add_argument("--model_name", type=str, default=None,
                   help="Specific model name (optional)")
parser.add_argument("--api_key", type=str, default="",
                   help="API key for OpenAI, TogetherAI, Anthropic or HuggingFace")
parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                   help="Base URL for OpenAI API (ignored for TogetherAI)")
parser.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Device for local models (llama/qwen)")
parser.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace token for private models")
parser.add_argument("--temperatures", type=str, default="1.0",
                   help="Comma-separated list of temperature values to test (e.g., '0.0,0.5,1.0,1.5')")

args = parser.parse_args()

# 모델별 기본 설정
model_defaults = {
    "openai": {"model_name": "gpt-4.1"},
    "claude": {"model_name": "claude-sonnet-4-5"},
    "llama": {"model_name": "meta-llama/Llama-3.1-8B-Instruct"},
    "qwen": {"model_name": "Qwen/Qwen2.5-32B-Instruct"}
}

# 모델 이름이 지정되지 않은 경우 기본값 사용
if args.model_name is None:
    args.model_name = model_defaults[args.model_type]["model_name"]

# MASEval 인스턴스 생성
if args.model_type == "openai":
    evaluator = MASEval(
        model_type="openai",
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name
    )
elif args.model_type == "together":
    evaluator = MASEval(
        model_type="together",
        api_key=args.api_key,
        model_name=args.model_name
    )
elif args.model_type == "claude":
    evaluator = MASEval(
        model_type="claude",
        api_key=args.api_key,
        model_name=args.model_name
    )
elif args.model_type in ["llama", "qwen"]:
    evaluator = MASEval(
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.device,
        hf_token=args.hf_token
    )

# 모델별 폴더명 생성
model_folder = get_model_folder_name(args.model_type, args.model_name)

# Temperature 값들 파싱
temperatures = [float(t.strip()) for t in args.temperatures.split(',')]

seed = args.seed
random.seed(seed)
np.random.seed(seed)

input_path = f"annotated/1/{args.dataset}"

print(f"Seed {seed} Running single file evaluation with {args.model_type} model...")
print(f"Model: {args.model_name}")
print(f"Method: {args.method}")
print(f"Input: {input_path}")
print(f"Temperatures: {temperatures}")
print(f"GT: {args.gt}")
print("-" * 80)

# 각 temperature 값에 대해 반복 실행
for temp in temperatures:
    # Temperature 설정
    evaluator.temperature = temp
    
    # Temperature별 출력 경로 생성
    if temp == 1.0:
        output_path = f"results/{model_folder}/{args.method}/seed_{seed}/{args.dataset}"
    else:
        output_path = f"results/{model_folder}/{args.method}/seed_{seed}/temp_{temp}/{args.dataset}"
    
    print(f"\nRunning with temperature={temp}...")
    print(f"Output: {output_path}")
    
    try:
        evaluator.evaluate_file(file_path=input_path, method=args.method, output_path=output_path, gt=args.gt)
        print(f"✓ Completed temperature={temp}")
    except Exception as e:
        print(f"✗ Error with temperature={temp}: {e}")
        continue
    
    print("-" * 80)

print("\nAll temperature evaluations completed!")