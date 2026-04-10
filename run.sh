export HF_TOKEN=YOUR_HF_TOKEN
# export TRANSFORMERS_VERBOSITY=info
export CUDA_VISIBLE_DEVICES=$6
export TOGETHER_NO_BANNER=1

model_type=$1 # openai, together, claude, qwen
model_name=$2 # gpt-4.1, gpt-5.1, o3-mini, openai/gpt-oss-120b, claude-sonnet-4-5, Qwen/Qwen3-8B
method=$3
seed=$4
dataset=${5:-manual} # manual, automatic

if [ "$dataset" = "manual" ]; then
    for file in $(seq 1 58); do
        python run_maseval.py --model_type $model_type --model_name $model_name --method $method --dataset $dataset/$file.json --seed $seed
    done
elif [ "$dataset" = "automatic" ]; then
    for file in $(seq 1 126); do
        python run_maseval.py --model_type $model_type --model_name $model_name --method $method --dataset $dataset/$file.json --seed $seed
    done
else
    echo "Invalid dataset"
    exit 1
fi