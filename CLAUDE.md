# ChartVCR Project

Chart-Verifiable Causal Rewards for Chart Reasoning via GRPO.

## 핵심 가이드 문서

- **`docs/quickstart_guide.md`** — 학습/추론 최적 환경 및 즉시 실행 가이드. 환경, TRL 패치, GRPO 학습 커맨드, 하이퍼파라미터, OOM 방지 설정 포함.
- **`docs/experiment_log.md`** — 환경/DeepSpeed/GPU 조합별 시도 결과 전체 기록. OOM 케이스, 디버깅 히스토리, 좀비 프로세스 대응 포함.

## 환경

- **Python**: `/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python` (유일하게 동작하는 환경)
- torch 2.10.0+cu128, vLLM 0.17.1rc1, transformers 5.3.0.dev0, TRL 0.29.0 (패치됨)
- **TRL 재설치 금지** — 패치가 사라짐

## GPU 정책

- GPUs 2-11: 사용 가능 (GRPO 학습 + vLLM)
- GPUs 0-1, 12-15: 유저 허가 시에만 사용
- **절대 `nvidia-smi --gpu-reset` 실행 금지** — 다른 사용자 프로세스에 영향. zombie GPU memory 발생 시 유저에게 PID 기반 kill 커맨드를 제공할 것.
- **절대 `fuser -k /dev/nvidia*` 실행 금지**
- 프로세스 kill은 반드시 PID 기반으로만 (`kill <PID>`)
- 좀비 GPU 메모리 발생 시 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`로 PID 확인 후 유저에게 `kill -9 <PID>` 커맨드 제공. 직접 실행하지 않음.
- 다른 사용자 GPU 프로세스 (GPU 8-15 등) 절대 건들지 않기

## Eval (벤치마크 추론)

### 원칙

- **가용 GPU 전부 사용**: 각 GPU에 TP=1 vLLM 서버를 띄워 throughput 극대화
- **비동기 병렬 요청**: 서버당 최대 5 concurrent 요청, 라운드로빈 로드밸런싱
- **샘플 단위 append 저장**: JSONL에 샘플 완료 즉시 append → 중단 후 resume 가능
- **resume 지원**: 이미 완료된 sample_id는 자동 skip
- **Qwen3.5 thinking mode**: `--reasoning-parser qwen3` + `enable_thinking: True`

### vLLM 서버 시작

```bash
# 가용 GPU 전부에 TP=1 서버 (4B 모델은 TP=1만 가능)
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
        --model <MODEL_PATH> \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --port $port \
        --trust-remote-code \
        --reasoning-parser qwen3 \
        > /tmp/vllm_eval_$i.log 2>&1 &
done
```

### Eval 실행

```bash
python eval_multi_server.py \
    "8000,8001,...,8011" \
    "<MODEL_PATH>" \
    "results/v7/<run_name>" \
    "chartqa_human,chartqa_augmented,charxiv_reasoning,chartqa_pro,chartmuseum"
```

- **스크립트**: `eval_multi_server.py` — 라운드로빈 분산, 서버당 5 concurrent, 샘플 단위 append
- **벤치마크**: chartqa_human, chartqa_augmented, charxiv_reasoning, chartqa_pro, chartmuseum
- **결과**: `results/v7/<run_name>/<benchmark>.jsonl`

## Training (GRPO + LoRA)

### 원칙

- **메모리 최소화 기법 필수 적용**:
  - `gradient_checkpointing=True` + `gradient_checkpointing_kwargs={"use_reentrant": True}` (Qwen3.5 호환)
  - `use_liger_kernel=True` (fused kernels로 activation 메모리 절감)
  - `attn_implementation="flash_attention_2"`
- **DeepSpeed ZeRO-3 (CPU offloading 없음)** 선호 — offloading은 속도 저하가 큼
- **batch size 통제**: 실험 간 effective batch size를 동일하게 유지해야 함
  - `effective_batch = num_gpus × per_device_batch × gradient_accumulation_steps`
  - target batch size에 도달한 후에는 GPU를 더 늘려도 batch를 늘리지 않음
  - `gradient_accumulation_steps=1`로 target batch 도달 시 더 이상 GPU 추가 불필요
- **가용 GPU 활용**: 메모리 기법으로 GPU당 부담을 줄이고, 필요한 만큼만 사용

### 스크립트

```bash
# vLLM 서버 (generation용, 별도 GPU에 trl vllm-serve)
CUDA_VISIBLE_DEVICES=<GPU> trl vllm-serve \
    --model <MODEL_PATH> --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 --max_model_len 8192 \
    --port 9100 --trust_remote_code

# GRPO 학습
CUDA_VISIBLE_DEVICES=<TRAIN_GPUS> accelerate launch \
    --num_processes <N> --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_grpo_dapo.py \
    --reward_type <outcome_only|conditional_cvr> \
    --data data/charts_v2/chartvr_train_final.jsonl \
    --output_dir ckpt/<run_name> \
    --use_lora --lora_rank 64 --lora_alpha 128
```

### LoRA 머지 (eval 전)

```bash
python scripts/merge_lora.py --base models/qwen3.5-4b --lora ckpt/<run_name> --output ckpt/<run_name>_merged
```
