export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve \
    "Qwen/Qwen2.5-MATH-7B-Instruct" \
    --served-model-name "Qwen2.5-MATH-7B-Instruct" \
    --port 8010 \
    --tensor-parallel-size 4 \
    --dtype auto \
    --api-key "token-abc123" \
    --enable-prefix-caching
