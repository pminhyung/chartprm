# ChartVCR 학습/추론 빠른 시작 가이드

> 이 문서만 읽으면 시행착오 없이 바로 학습 및 추론을 시작할 수 있습니다.

## 환경

```bash
# 유일하게 동작하는 환경 (변경 금지)
export PYTHON=/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python
export ACCELERATE=/ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate
export TRL=/ex_disk2/mhpark/poc/vllm_nightly_env/bin/trl

# 버전: torch 2.10.0+cu128, vLLM 0.17.1rc1, transformers 5.3.0.dev0, TRL 0.29.0, flash_attn 2.8.3
# 다른 환경 (vllm_env, chartvr_env, base conda) 사용 금지 — ABI/버전 충돌
```

## 필수 TRL 패치 (이미 적용됨)

아래 파일들이 이미 수정되어 있음. **TRL을 재설치하면 패치가 사라지므로 pip install trl 금지.**

1. `trl/scripts/vllm_serve.py` — SamplingParams 호환성, logprobs None 처리
2. `trl/generation/vllm_generation.py` — weight sync 비활성화
3. `trl/trainer/grpo_trainer.py` — mm_token_type_ids VLM 호환성

## 추론 (Zero-shot 평가)

```bash
# 모델: Qwen3-VL-8B-Thinking
# 핵심: system prompt로 <answer> 태그 강제, enable_thinking=True
cd /ex_disk2/mhpark/poc/chartvr

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON eval_all.py \
    --model qwen3vl_zeroshot \
    --benchmark chartqa_human \
    --gpu-ids "0,1,2,3"

# 벤치마크: chartqa_human, chartqa_augmented, charxiv_reasoning, chartqa_pro
# GPU: TP=2 이상 필요 (8B 모델), TP=4 권장
# 결과: results/v4/ 디렉토리에 저장
```

## GRPO 학습

### 아키텍처
```
GPUs 0-1: vLLM 서버 (Qwen3-VL-8B-Thinking, TP=2, port 9100)
GPUs 2-11: GRPO 학습 (DeepSpeed ZeRO-3, 10 프로세스)
GPUs 12-15: 사용 금지 (다른 사용자)
```

### Step 1: vLLM 서버 시작
```bash
CUDA_VISIBLE_DEVICES=0,1 $TRL vllm-serve \
    --model models/qwen3vl-8b-thinking \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 8192 \
    --port 9100 \
    --trust_remote_code \
    > logs/vllm_server.log 2>&1 &

# Health check (약 90초 소요)
until curl -s http://localhost:9100/health/ > /dev/null 2>&1; do sleep 5; done
echo "vLLM HEALTHY"
```

### Step 2: GRPO 학습 시작
```bash
# DeepSpeed ZeRO-3 config (CPU offload 없음)
cat > /tmp/ds_z3.json << 'EOF'
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
EOF

cat > /tmp/accel.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 10
deepspeed_config:
  deepspeed_config_file: /tmp/ds_z3.json
EOF

# Baseline 학습
CHARTVR_ROOT=/ex_disk2/mhpark/poc/chartvr \
PYTHONPATH=/ex_disk2/mhpark/poc/chartvr \
VLLM_PORT=9100 \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9,10,11 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$ACCELERATE launch --config_file /tmp/accel.yaml \
    train_grpo.py --reward_type outcome_only --output_dir ckpt/baseline

# ChartVCR 학습 (reward만 다름)
# train_grpo.py --reward_type chartvr --output_dir ckpt/cvr
```

### 핵심 하이퍼파라미터 (OOM 방지)

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| `per_device_train_batch_size` | **1** | 8B VLM forward 메모리 |
| `num_generations` | **2** | vLLM 서버 부하 + 메모리 |
| `max_completion_length` | **1024** | 2048은 OOM |
| `generation_batch_size` | **num_processes와 동일** (10) | TRL 요구사항 |
| `gradient_accumulation_steps` | **8** | effective batch = 80 |
| DeepSpeed | **ZeRO-3 (offload 없음)** | ZeRO-2는 OOM, CPU offload는 너무 느림 |

### GPU 메모리 사용량 (실측)

| 설정 | GPU당 메모리 | step 시간 | 결과 |
|------|------------|----------|------|
| ZeRO-2, 8 GPU, batch=1 | 36.5GB | — | ❌ OOM (optimizer step) |
| ZeRO-3 + CPU offload, 8 GPU, batch=1 | 12GB | 98s (너무 느림) | ✅ 동작하나 비실용적 |
| **ZeRO-3 (offload 없음), 10 GPU, batch=1** | **27GB** | **97s** | **✅ 정상 동작, reward=0.77** |

## 데이터셋

```bash
# GRPO 데이터셋 (캐시, 28299 samples with PIL images)
ls data/grpo_dataset_v4/

# 평가 데이터
ls data/chartqa/test/      # ChartQA Human + Augmented
ls data/charxiv/           # CharXiv
ls data/chartqa_pro/       # ChartQA-Pro
```

## 주의사항

1. **GPU 12-15 절대 사용 금지** (다른 사용자)
2. **TRL 재설치 금지** (패치 사라짐)
3. **`sudo fuser -k /dev/nvidia*` 금지** — 반드시 개별 GPU 번호 지정
4. **vLLM 서버 crash 시 좀비 발생** — `sudo fuser -k /dev/nvidiaN`으로 개별 해제
5. **vLLM 서버와 학습은 반드시 분리된 GPU 사용** (colocate mode 불가)
6. **`generation_batch_size`는 반드시 `num_processes * per_device_train_batch_size`와 같거나 나눠떨어져야 함**
