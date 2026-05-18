# ChartVCR Project

Chart-Verifiable Causal Rewards for Chart Reasoning via GRPO.

## 응답 형식 (모든 답변 필수)

모든 답변은 아래 **4-블록 보고 형식**을 따른다. 분석/디버깅/구현/조사 — 어떤 작업이든 동일.

### 필수 4블록 (이 순서, 이 헤더 그대로 사용)

**맥락**: 1–2줄. 상위 목표 + 최종 산출물.
**이 스탭**: 1–2줄. 이 작업이 전체에서 무엇을 해소하는지 + 통과/완료 기준.
**발견**: 3–6 bullet, 각 1줄. 사실/숫자/파일경로 위주. 정량 주장은 반드시 숫자. 결론 먼저 → 근거 한 줄.
**액션**: 1–3 bullet. 구체 수정/검증 항목. 의사결정 필요 시 옵션 나열 후 **(추천)** 1개 + 근거 1줄. 마지막 줄에 유저 결정 필요 사항을 `결정 필요:` 로 명시.

### 금지 (위반 시 답변 폐기 후 재작성)

- ASCII 박스 표 (`┌─┐`, `├─┤`, `└─┘`) — 절대 금지. 표가 꼭 필요하면 markdown pipe table, **3열 × 4행 이하**.
- 같은 사실을 표 + 문장으로 중복 기술
- "본 보고서는…", "위와 같이 정리하면…", "분석한 결과…" 같은 메타·자기서술 문장
- 결론 없는 나열, 액션 없는 분석, 숫자 없는 정량 주장
- 코드/설정 dump — `file:line` 인용 + 핵심 5줄 이내 발췌만

### 길이·톤

- 일반 답변 **25줄 이내**, 깊은 분석 **50줄 이내**. 4블록 구조는 동일.
- 표 비중 본문의 30% 이하.
- 사실문으로. 불확실 시 `[추정]` `[미확인]` 태그 명시.
- 코드 인용은 ` ``` ` fenced + path:line 헤더 1줄 + 5줄 이내.

### Pedagogical mode — when reporting analysis / experimental results

**Trigger** (auto-engage when user signals lack of prior context):
- explicit: "단계적", "차근차근", "쉽게", "다 읽지 않았어", "explain step by step"
- implicit: user asks "왜 …", "어디가 문제", or shows they're new to the result

**Expand the `발견` block into 4 labeled mini-sections** (other 3 blocks unchanged):

1. **Hypothesis** — what we originally believed and why. Tie to the research goal in one line. Always name the assumption being tested.
2. **Measurement** — what we set up to test it. For every metric, append `(what passing it would prove)` so the reader knows why the number matters before they see it.
3. **Result** — raw numbers per metric, with PASS/FAIL vs target. Use the same metric names as §Measurement. Numbers only, no narrative here.
4. **Diagnosis** — where the expectation broke. Include **2–3 verbatim data examples** (sample id + claim + gold + pred) before the conclusion. End with one line: `Hypothesis: X. Reality: Y. So {signal/method} is unviable for {goal}.`

**Hard rules**:
- Never dump metric names without explaining them on first appearance.
- Never cite a result number without referencing the metric it came from in §Measurement.
- No jargon that wasn't introduced in the same response (or `MEMORY.md`).
- 50-line cap still applies (deep analysis).
- 4-block frame (`맥락 / 이 스탭 / 발견 / 액션`) is non-negotiable — the expansion lives inside `발견`.
- `액션` still ends with `결정 필요:` line.

### 적용 예시 (이 형태로 답변할 것)

```
**맥락**: v8 eval 결과 신뢰도 검증 / 산출물: extraction fix + v7↔v8 fair 비교.
**이 스탭**: empty content fallback 오류 원인 규명. 통과 기준: Row 4 AVG 가 v7_row_b 와 동일 stop 조건에서 비교 가능.
**발견**:
- v7(stop 없음) vs v8(stop=`</answer>`) 파이프라인 차이로 content 포맷 비대칭.
- empty 원인 3종: (A) think runaway 30%, (B) prompt 예시 echo, (C) 조기 commit 50%+ — (C)는 `reasoning` 끝에 정답 생존.
- `rescore_v5.py:reconstruct` 가 (C) 케이스 84/121건(69.4%) 죽임 — v5 자체 버그.
**액션**:
- (추천) F1+F2 = v6 extraction: `rescore_eval_v6.py` reconstruction 제거 + `extraction.py` 예시 mask. 5분, Row 3 학습 무관.
- (중기) F3 prompt 추상화 + F4 v7 재평가는 paper main table 확정 시 일괄.
**결정 필요**: v6 구현 진행할까요? (Y → F1+F2 즉시 패치)
```

## 핵심 가이드 문서

- **`docs/training_memory_reference.md`** — **학습 세팅 필수 참조**. 모델별/GPU별 메모리 테스트 결과, 추천 설정, OOM 방지. 학습 시 이 문서의 설정을 그대로 사용하거나 `train_grpo_dapo.py`의 현재 코드를 사용할 것.
- **`docs/quickstart_guide.md`** — 환경 및 즉시 실행 가이드.
- **`docs/experiment_log.md`** — 환경/DeepSpeed/GPU 조합별 시도 결과 전체 기록.

## 환경

- **Python**: `/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python` (유일하게 동작하는 환경)
- torch 2.10.0+cu128, vLLM 0.17.1rc1, transformers 5.3.0.dev0, TRL 0.29.0 (패치됨)
- **TRL 재설치 금지** — 패치가 사라짐

## GPU 정책

- **가용 GPU 전부 사용 가능** (0-15 제한 없음). `nvidia-smi`로 비어있는 GPU 확인 후 사용.
- **절대 `nvidia-smi --gpu-reset` 실행 금지** — 다른 사용자 프로세스에 영향. zombie GPU memory 발생 시 유저에게 PID 기반 kill 커맨드를 제공할 것.
- **절대 `fuser -k /dev/nvidia*` 실행 금지**
- 프로세스 kill은 반드시 PID 기반으로만 (`kill <PID>`)
- 좀비 GPU 메모리 발생 시 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`로 PID 확인 후 유저에게 `kill -9 <PID>` 커맨드 제공. 직접 실행하지 않음.
- 다른 사용자 PID의 프로세스는 절대 kill하지 않음 — 메모리 사용 중인 GPU 회피

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

## QA Generation (397B)

397B (Qwen3.5-397B-A17B-FP8)은 QA 생성·CoT·reward verifier 공용.

### 원칙

- **반드시 `scripts/launch_397b_vllm.sh` 사용** — 추천 플래그가 전부 default로 박혀 있음
  - `--reasoning-parser qwen3` — `<think>...</think>` 자동 파싱, 누락 시 thinking 모드에서 raw 태그 노출 + artifacts
  - `--generation-config vllm` — 모델의 `generation_config.json` (thinking 기본값 temp=0.6/top_p=0.95) 무시. `chartvr/llm_client.py`가 `enable_thinking` 플래그 기반으로 올바른 샘플링 파라미터를 명시 주입하므로 generation_config 오염 방지
  - `--enforce-eager` — GPTQ-int4 + TP=8 안정화
  - `setsid ... </dev/null & disown` — 세션 compaction SIGHUP 회피
- **샘플링 파라미터**: `chartvr/config.py::SAMPLING_PARAMS["397b"]` (thinking/instruct 분리)
  - instruct (non-thinking): temp=0.7, top_p=0.8, top_k=20, min_p=0 (Qwen 공식 권장)
  - thinking: temp=0.6, top_p=0.95, top_k=20, min_p=0
  - `MultiHostClient.chat()`이 `enable_thinking`에 따라 자동 주입
- **Dual-host multi-port**: 9200 (GPU 0-7, 4k) + 9201 (GPU 8-15, 64k) 병렬 운용

### 서버 시작

```bash
# 표준 dual-host (QA 생성 throughput 2×)
./scripts/launch_397b_vllm.sh 9200 0,1,2,3,4,5,6,7 4096
./scripts/launch_397b_vllm.sh 9201 8,9,10,11,12,13,14,15 65536

# Ready 확인
curl -s http://localhost:9200/v1/models | jq
curl -s http://localhost:9201/v1/models | jq
```

### QA 생성 실행

```bash
# 4개 sub-prompt × dual-host × concurrent=6 (host당 6, total 12)
HOSTS="http://localhost:9200/v1,http://localhost:9201/v1"
for qa_type in sci_ranking sci_numeric sci_trend sci_compare; do
    python scripts/generate_qa.py generate \
        --eligible data/charts_v9/block_c_eligible_v2_${qa_type#sci_}.json \
        --prompt ${qa_type} \
        --n_qa_per_chart 1 \
        --hosts "$HOSTS" \
        --max_concurrent_per_host 6 \
        --output data/charts_v9/block_c_api_qa_v2_${qa_type}.jsonl
done
```

### GPU 운영 정책 (v9.1+)

- **QA 생성 단계**: 9200 + 9201 둘 다 사용 (throughput 2×)
- **학습 전환 단계**: **9200만 kill**하여 GPU 0-7 확보. 9201 은 그대로 유지.
- 학습: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

## Training (GRPO + LoRA)

### vLLM Throughput Recipe (2026-04-28~)

채택 config: **vLLM TP=4 (GPU 0-3) + train 4-rank ZeRO-3 (GPU 4-7)**, `effective_batch=32` 고정. step time 200s→139s (−30%).

- 상세 recipe + 시도 실패 옵션: **`docs/reference/grpo_vllm_throughput.md`**
- DP는 dense 모델에서 거부됨 (재시도 금지) — TP=N 사용
- vLLM EngineCore 종료 확인은 `ps -eo pid,cmd | grep -iE 'vllm|EngineCore'` (pgrep으로는 자식 안 잡힘)
- Resume: `--resume_from_checkpoint ckpt/<row>/checkpoint-<N>` 지원

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
