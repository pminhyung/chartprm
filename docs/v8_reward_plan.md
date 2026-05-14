# Plan: Reward 체계 구조적 한계 분석 + v3 이후 전략

## Status: Row 4b (reward v3, multiplicative) 학습 중 (step 193/1631)

---

## Part A: v3 수정 (이미 구현 완료, 학습 진행 중)

v2의 additive inflation 문제를 multiplicative로 해결:
```
v2: min(1.0, r_acc + 0.3 * r_proc)     → 26.4% wrong at 1.0, gap 0.273
v3: r_acc + 0.3 * (1-r_acc) * r_proc_v2 → 0% wrong at 1.0, gap 0.713
```
+ process reward v2: grounding 10% tolerance, arithmetic/CSV default 0.0

---

## Part B: 구조적 한계 분석 (v3로도 해결 안 되는 문제들)

### 한계 1: Reward Signal 범위 불일치 (가장 심각)

**전제**: 모델 입력은 학습/평가 모두 동일 (이미지 + 질문). CSV는 모델에 노출되지 않음.
CSV는 오직 reward 함수 내부에서 process reward 계산 시에만 사용.

| | GRPO 학습 답변 | CQA-H 답변 | CharXiv 답변 | ChartMuseum 답변 |
|---|---|---|---|---|
| 수치 | **100%** | 68% | 45% | 27% |
| 텍스트 | **0%** | 32% | 55% | 73% |

**문제**: Process reward는 수치 답변에서만 continuous signal (grounding + arithmetic).
텍스트 답변은 `relaxed_text_match`(binary 0/1)만 받음 → process reward signal 없음.
학습 데이터가 100% 수치이므로 **모든 학습 샘플에서 process reward가 작동하지만**,
평가에서 텍스트 비중이 높은 벤치마크일수록 process reward 학습의 transfer가 제한적.

### 한계 2: Process Reward Floor Effect

분석된 100개 학습 샘플에서:
- r_proc ≥ 0.95: **79%** (거의 모두 만점)
- r_proc 0.5-0.95: 16%
- r_proc < 0.5: 5%

**원인**: Grounding accuracy 97% (CSV 숫자가 reasoning에 거의 다 등장) → r_proc 분산 부족.
v3 multiplicative fix가 이 floor effect을 완화하지만, r_proc 자체의 분산이 작으면 여전히 효과 제한적.

### 한계 3: Rule-Based Verifier가 잡지 못하는 에러 (60%)

| 잡을 수 있음 (40%) | 잡을 수 없음 (60%) |
|---|---|
| 산술 오류 (22%) | 의미 오해: "growth" vs "decline" |
| 값 매칭 실패 | 질문 오독: "average" vs "total" |
| 계산 체인 오류 | 엔티티 혼동: Tunisia vs Turkey |
| 컬럼 조회 오류 | 시간 오류: 2020 vs 2019 |

Wrong answer의 54%가 logic error → rule verifier로는 원천적으로 구별 불가.

### 한계 4: 과도한 Reasoning 길이

| | Correct | Wrong | 배율 |
|---|---|---|---|
| CQA-H | 197 tokens | 859 tokens | 4.3x |
| CharXiv | 443 tokens | 1,414 tokens | 3.2x |
| ChartMuseum | 520 tokens | 1,517 tokens | 2.9x |

모델이 확신 없는 문제에서 reasoning을 3-5배 길게 쓰지만, 현재 reward에는 길이 패널티가 없음.
(soft_overlong_punishment은 max_completion_length 기준이지 reasoning 길이 기준이 아님)

### 한계 5: Format Compliance 하락

| Benchmark | `<answer>` 태그 사용률 | Accuracy |
|---|---|---|
| CQA-H (easy) | 93% | 84% |
| CharXiv (medium) | 67% | 45% |
| ChartMuseum (hard) | 53% | 30% |

어려운 벤치마크일수록 format compliance 하락 → 답 추출 실패 → 정확도 추가 하락.

---

## Part C: 한계별 영향도 + 해결 가능성

| 한계 | 영향도 | v3로 해결? | 추가 대응 필요 여부 |
|------|--------|-----------|-------------------|
| 1. Train-eval 분포 불일치 | **매우 높음** | ❌ | GRPO 데이터에 텍스트 답변 추가 필요 |
| 2. r_proc floor effect | 높음 | △ 부분적 (tolerance 강화) | 데이터 난이도↑ 또는 verifier 강화 |
| 3. Verifier 60% 미검출 | 중간 | ❌ 구조적 한계 | 질문-답 타입 일관성 체크 추가 가능 |
| 4. Reasoning 길이 폭발 | 중간 | ❌ | 별도 length penalty 검토 |
| 5. Format compliance 하락 | 낮음 | ❌ | SFT에서 이미 학습, GRPO 영향 제한적 |

**핵심 인사이트**: v3는 "공식 설계 오류"를 수정하지만, "무엇을 보상하는가"의 범위(coverage)는 바꾸지 않음.
Process reward가 100% numeric+CSV에서만 작동하는 한, 텍스트 답변 비중이 높은 벤치마크에서의 개선은 제한적.

→ Row 4b 결과 확인 후 유저와 다음 전략 논의 예정.

---

## Part D: 학습 과정 트러블슈팅 이력

### 이미지 해상도 문제 (Qwen VL token mismatch)

TRL GRPOTrainer에서 Qwen3.5-4B VLM 학습 시, 배치 내 이미지 크기가 다르면
`ValueError: Image features and image tokens do not match` 발생.

| 시도 | 설정 | 결과 |
|------|------|------|
| 1차 | 원본 크기 (max 1024 resize) | step ~110에서 crash (tokens:559, features:558) |
| 2차 | 28px 배수 정렬 | step ~110에서 crash (동일 에러) |
| 3차 | 32px 배수 정렬 | step ~119에서 crash (tokens:560, features:558) |
| 4차 | **960×960 고정** | Row 3: 978 steps 완주 ✅, Row 4: 1224 steps 완주 ✅, **Row 4b: step 148 crash** (tokens:901, features:900) |
| 5차 | **448×448 고정** | Row 4b: step 193+ 통과, 안정 ✅ |

**원인**: Qwen VL의 내부 이미지 tiling은 이미지 크기에 따라 동적으로 tile 수를 결정.
배치 내 이미지들의 tile 수가 다르면 padding 과정에서 token/feature 수 불일치 발생.
동일 크기라도 비결정적으로 발생할 수 있어, 960×960은 일부 시드에서만 안정.

**최종 해결**: 448×448 (patch_size=16, 448/16=28, merge_size=2 → 항상 784 visual tokens).
이 크기는 Qwen VL의 최소 tiling 단위와 정확히 일치하여 tile 분할이 발생하지 않음.

**평가에는 영향 없음**: eval은 vLLM이 이미지를 자체 처리하므로 토큰 불일치 미발생.
학습 시 448×448로 축소해도 차트의 핵심 정보(숫자, 라벨, 추세)는 보존됨.

### SFT 학습 이슈

| 이슈 | 원인 | 해결 |
|------|------|------|
| `SFTConfig: unexpected keyword 'max_seq_length'` | TRL 버전 차이 | `max_seq_length` → `max_length`로 변경 |
| GPU 8-9 충돌 | 다른 사용자 프로세스 점유 | GPU 2-7만 사용 (6 GPUs) |

### GRPO 학습 이슈

| 이슈 | 원인 | 해결 |
|------|------|------|
| `generation_batch_size must be divisible by num_generations` | 5 GPUs × 1 batch = 5, num_gen=4 | `_lcm(num_gen, global_batch)` 적용 |
| `generation_batch_size must be divisible by global batch size` | 동일 원인의 역방향 | LCM으로 양방향 정렬 |
| TRL vLLM client 404 | 일반 vLLM 서버 사용 → TRL 전용 endpoint 없음 | `trl vllm-serve` 사용 |
| vLLM model name mismatch | `MultiHostClient`가 397B 모델명 사용 | `--model` 파라미터 명시 |
| GPU zombie 프로세스 | TRL vLLM 종료 후 GPU 메모리 미해제 | PID 기반 kill 필요 (GPU 2,3에 잔존) |

## Current Status
- **Row 4b**: step 193/1631, conditional_v3, 학습 진행 중 (ETA ~15h)
- 모니터링: 8h check, 18h completion check 설정됨

## Files
- `code/rewards/rule_verifier_fast.py` — compute_process_reward_v2 (구현 완료)
- `train_grpo_dapo.py` — reward_conditional_v3 (구현 완료), 이미지 448x448
- `data/grpo_v2.jsonl` — 현재 100% numeric, 텍스트 미포함
- `docs/reward_v3_diagnosis.md` — 상세 진단 보고서
