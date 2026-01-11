base_score_path=./selected_frames/longvideobench/blip
score_type=selected_frames
dataset_name=longvideobench

# Set NUMEXPR_MAX_THREADS to avoid warnings
export NUMEXPR_MAX_THREADS=64

# Set HuggingFace to offline mode to avoid network connection errors
# This forces transformers to use only local files
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Alternative: use HuggingFace mirror if available
# export HF_ENDPOINT=https://hf-mirror.com

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name 

# Get absolute path to the model
# Script is in evaluation/scripts/llava_video_7b/, so go up 3 levels to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/checkpoints/llava_video_7b/LLaVA-Video-7B-Qwen2"

# Verify model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=$MODEL_PATH,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False,use_topk=True \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_7b_qwen_lvb_v \
    --output_path ./results/${score_type}
