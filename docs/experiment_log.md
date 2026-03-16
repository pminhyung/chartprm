# ChartVCR 실험 환경 세팅 로그

## 1. 환경별 시도 결과

### 1.1 Python 환경

| 환경 | Python | torch | vLLM | transformers | flash_attn | TRL | 결과 |
|------|--------|-------|------|-------------|------------|-----|------|
| base conda | 3.10 | 2.5.1+cu121 | 0.10.0 | 4.54.1 | ❌ ABI mismatch | — | vLLM import 실패 |
| base conda (fixed) | 3.10 | 2.5.1+cu121 | 0.7.3 | 4.54.1 | ❌ | 0.16.0 | vLLM OK, Qwen3-VL 미지원 |
| vllm_env | 3.10 | 2.9.1+cu128 | 0.16.0 | 4.54.1→5.3.0 | ❌ ABI mismatch | 0.29.0 | flash_attn 빌드 실패 |
| chartvr_env | 3.10 | 2.5.1+cu124→2.6.0 | 0.7.3→0.8.5 | 5.3.0 | ❌ | 0.16.1 | 버전 충돌 반복 |
| **vllm_nightly_env** ✅ | **3.10** | **2.10.0+cu128** | **0.17.1rc1** | **5.3.0.dev0** | **2.8.3 ✅** | **0.29.0** | **모든 기능 정상** |

### 1.2 vllm_nightly_env에 필요한 TRL 패치 목록

| 파일 | 패치 내용 | 이유 |
|------|-----------|------|
| `trl/scripts/vllm_serve.py` | `truncate_prompt_tokens` 제거 | vLLM 0.17.1의 SamplingParams에 없는 파라미터 |
| `trl/scripts/vllm_serve.py` | `SamplingParams` inspect filter 제거 | `**kwargs` 시그니처로 `n` 파라미터가 필터링됨 |
| `trl/scripts/vllm_serve.py` | logprobs/logprob_token_ids None→[] | 응답 validation 에러 방지 |
| `trl/generation/vllm_generation.py` | `init_communicator` 주석 처리 | NCCL cross-process 충돌 방지 |
| `trl/generation/vllm_generation.py` | `update_named_param`→pass | weight sync 비활성화 (서버 모드) |
| `trl/generation/vllm_generation.py` | `reset_prefix_cache`→pass | communicator 없이 호출 시 에러 |
| `trl/trainer/grpo_trainer.py` | `mm_token_type_ids` 확장 + 길이 정렬 | VLM 이미지 토큰 확장 시 mask 불일치 해결 |

---

## 2. GRPO 학습 DeepSpeed + GPU 조합 시도

### 2.1 BigCharts-R1 Custom Trainer (폐기)

| GPUs | DeepSpeed | batch | max_comp | 결과 |
|------|-----------|-------|----------|------|
| 8 (base) | ZeRO-3 | 1 | 2048 | cuDNN 에러 (torch 2.5.1+cu121) |
| 8 (vllm_env) | ZeRO-3 | 1 | 1024 | `IndexError: mask shape mismatch` (VLM 토큰 확장) |
| 2 | ZeRO-3 | 1 | 256 | OOM (2 GPU로 8B model forward 불가) |

### 2.2 TRL GRPOTrainer — Colocate Mode

| GPUs | DeepSpeed | vllm_gpu_util | max_model_len | 결과 |
|------|-----------|---------------|---------------|------|
| 4 | ZeRO-2 | 0.3 | 4096 | `No available memory for KV cache` |
| 4 | ZeRO-3 | 0.3 | 4096 | `No available memory for KV cache` |
| 4 | ZeRO-2 | 0.5 | 4096 | `max_model_len=262144 needs 36GB KV` |
| 8 | ZeRO-2 | 0.3 | 4096 | `No available memory for KV cache` |

**결론: Colocate mode는 8B VLM에서 OOM. 모델 2카피(학습+vLLM) 불가능.**

### 2.3 TRL GRPOTrainer — Server Mode (최종 성공)

| GPUs (train) | GPUs (vLLM) | DeepSpeed | batch | num_gen | max_comp | gen_batch | 결과 |
|-------------|-------------|-----------|-------|---------|----------|-----------|------|
| 8 (Z2) | 4 (TP=4) | ZeRO-2 | 1 | 8 | 2048 | 8 | `truncate_prompt_tokens` 에러 |
| 8 (Z2) | 4 (TP=4) | ZeRO-2 | 1 | 8 | 2048 | 8 | `logprobs=None` 에러 |
| 4 (Z2) | 2 (TP=2) | ZeRO-2 | 2 | 2 | 2048 | 2 | `n` 파라미터 필터링 → 1개만 생성 |
| 4 (Z2) | 2 (TP=2) | ZeRO-2 | 2 | 2 | 2048 | 8 | `mm_token_type_ids` mask 불일치 |
| 4 (Z2) | 2 (TP=2) | ZeRO-2 | 1 | 2 | 1024 | 8 | OOM (36.5GB/GPU, optimizer step에서) |
| 8 (Z2) | 2 (TP=2) | ZeRO-2 | 1 | 2 | 1024 | 8→10 | OOM (batch_size 불일치 후 수정해도 OOM) |
| 8 (Z3+CPU) | 2 (TP=2) | ZeRO-3+CPU offload | 1 | 2 | 1024 | 8 | ✅ **성공! loss=0.059, reward=0.77** (너무 느림) |
| **10 (Z3)** | **2 (TP=2)** | **ZeRO-3 (offload 없음)** | **1** | **2** | **1024** | **10** | **✅ 현재 진행 중** |

---

## 3. 추론(Inference) 환경 설정

### 3.1 Zero-shot 평가

| 환경 | 모델 | GPUs | max_tokens | system_prompt | 결과 |
|------|------|------|-----------|---------------|------|
| vllm_env (0.16.0) | Qwen3-VL-8B-Thinking | 4 (TP=4) | 2048 | 없음 | CQA-H: 68.56% (answer 추출 부정확) |
| vllm_nightly (0.17.1) | Qwen3-VL-8B-Thinking | 4 (TP=4) | 2048 | 없음 | CQA-H: 6.32% (답 추출 실패, 전문장 반환) |
| vllm_nightly (0.17.1) | Qwen3-VL-8B-Thinking | 4 (TP=4) | 2048 | `enable_thinking=True` | CQA-H: 62.24% (`<think>` 태그 없이 reasoning) |
| **vllm_nightly (0.17.1)** | **Qwen3-VL-8B-Thinking** | **4 (TP=4)** | **2048** | **`<answer>` 강제 system prompt** | **CQA-H: 80.64%, CQA-A: 91.12%** ✅ |

### 3.2 핵심 추론 설정

```python
SYSTEM_PROMPT = (
    "You are an expert chart analyst. "
    "After your reasoning, you MUST put your final answer inside <answer> and </answer> tags. "
    "The <answer> tag should contain ONLY the core numeric value or keyword."
)
# vLLM: LLM.chat() with chat_template_kwargs={"enable_thinking": True}
# max_tokens=2048, temperature=0, image resize max 1024px
```

---

## 4. 모델별 메모리/GPU 요구사항

| 모델 | 파라미터 | FP16 크기 | 최소 GPU (추론) | 최소 GPU (GRPO 학습) |
|------|---------|----------|----------------|-------------------|
| Qwen3-VL-8B-Thinking | 8.2B | ~16GB | 2 (TP=2) | 10 (ZeRO-3) + 2 (vLLM) |
| Qwen3-VL-4B-Instruct | 4.4B | ~9GB | 1 | — (verifier only) |
| Qwen2.5-VL-7B | 7.6B | ~15GB | 2 (TP=2) | 8 (ZeRO-2) — SFT만 |

---

## 5. 좀비 프로세스 관련

| 원인 | 빈도 | 해결방법 |
|------|------|---------|
| vLLM 서버 crash | 매우 높음 | `sudo fuser -k /dev/nvidiaN` (개별 GPU만!) |
| `setsid` + crash | 높음 | `setsid` 사용 금지 |
| `kill -9` (bare) | 중간 | process group kill 사용: `kill -- -$PGID` |
| `sudo fuser -k` 과다 | 높음 | **개별 GPU만 지정!** 전체 사용 시 다른 사용자 프로세스도 죽음 |

**중요: `sudo fuser -k /dev/nvidia*` 절대 금지! 반드시 개별 GPU 번호 지정.**
