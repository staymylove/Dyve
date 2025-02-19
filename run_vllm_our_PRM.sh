export CUDA_VISIBLE_DEVICES=4,5,6,7

vllm serve \
    "/root/deepseek-r1-14b-cot-math-reasoning-full" \
    --served-model-name "deepseek-r1-14b-cot-math-reasoning-full" \
    --port 8011 \
    --tensor-parallel-size 4 \
    --dtype auto \
    --api-key "token-abc123" \
    --enable-prefix-caching



