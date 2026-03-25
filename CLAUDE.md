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

### 빠른 Eval 방법: 멀티 서버 병렬

사용 가능한 GPU에 각각 vLLM 서버(TP=1)를 띄우고, 벤치마크별로 순차 진행하되 모든 서버에 분산 요청하여 throughput 극대화.

```bash
# Step 1: GPU 0-7에 8개 vLLM 서버 시작 (TP=1, 각각 다른 포트)
for i in 0 1 2 3 4 5 6 7; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
        --model <MODEL_PATH> \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --port $port \
        --trust-remote-code \
        > /tmp/vllm_eval_$i.log 2>&1 &
done

# Step 2: 전체 서버 ready 확인 (~3분 소요)
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
    until curl -s http://localhost:$port/v1/models | grep -q model; do sleep 2; done
done

# Step 3: 8개 서버에 분산 요청하며 eval 실행
python eval_multi_server.py \
    "8000,8001,8002,8003,8004,8005,8006,8007" \
    "<MODEL_PATH>" \
    "results/v6b/<run_name>" \
    "chartqa_human,chartqa_augmented,charxiv_reasoning,chartqa_pro"
```

- **스크립트**: `eval_multi_server.py` — 서버 URL 리스트를 받아 라운드로빈으로 분산, 서버당 4 concurrent 요청
- **벤치마크**: chartqa_human, chartqa_augmented, charxiv_reasoning, chartqa_pro
- **결과**: `results/v6b/<run_name>/<benchmark>.jsonl`
- **주의**: 4B 모델은 TP=1만 가능. TP=4는 모델이 작아서 실패함.
