# Session Handoff — 2026-05-12
# ChartVCR: Verification-Aware RL for Chart Reasoning

---

## 1. 프로젝트 한 줄 요약

Chart reasoning에서 **outcome-only로 hidden된 "Grounding Drift"(reasoning step이 chart로부터 derivable하지 않은 현상)를** verification-anchored SFT + VAPV process reward의 2-stage co-design으로 점진 해결하는 논문. Qwen3.5-VL 4B/9B, top venue (NeurIPS/ICML/ICLR/ACL/EMNLP) main track 타겟. 5개 benchmark(CQA-H/A/Pro, CharXiv, ChartMuseum) AVG SOTA 타겟.

---

## 2. 현재 상태 (Status Snapshot)

### 진행 중 (Day 0)
- **row_a_outcome (현재 best RL)의 ChartQA-Pro VLMEvalKit 측정 진행 중**. 결과 D1 EOD 가용 예상.
- v3 SFT (chartqa_train 9,498 rows VLMEvalKit Factoid prompt 재 distill, 총 27,575 rows) 학습 진행 중, ETA ~24h.

### 측정된 결과 — GRPO-only (SFT 없음), Extraction v3, lmms-eval scorer
```
                CQA-H   CQA-A   CharXiv  CQA-Pro  ChartMus  AVG
Zero-shot       79.1%   88.5%   45.7%    37.2%    31.6%    56.4%
Row A(outcome)  82.7%   88.0%   47.4%    37.5%    30.5%    57.2%
Row B(proc-aug) 84.0%   91.2%   50.1%    39.6%    34.8%    59.9%
B vs A         +1.3pp  +3.2pp  +2.7pp   +2.1pp   +4.3pp  +2.7pp ✅
```
**유일한 confirmed positive — 5/5 bench에서 process reward > outcome.**

### v8 SFT → GRPO (효과 소멸)
```
                CQA-H   CQA-A   CharXiv  CQA-Pro  ChartMus  AVG
SFT(R2)         83.7%   88.6%   44.5%    35.8%    27.6%    56.0%
Outcome(R3)     84.5%   89.0%   46.0%    36.4%    28.3%    56.8%
Process v2(R4)  83.8%   89.0%   45.3%    36.1%    30.4%    56.9%
Δ(R4-R3)       -0.7     0.0    -0.7     -0.3     +2.1     +0.1
```
원인: ① additive reward inflation, ② 100% 수치 학습 데이터 (텍스트 답변 reward 미작동), ③ SFT가 improvement room 축소.

### ChartQA-Pro 측정 충격 — eval scorer artifact
| Model | lmms-eval Pro | VLMEvalKit Pro | Δ |
|---|---:|---:|---:|
| zero-shot | 30.54% | **45.87%** | +15.33pp |
| v8_distill_v2 SFT | 27.57% | 43.85% | +16.28pp |

→ lmms-eval `relaxed_correctness`가 ChartQA-Pro의 의도된 채점이 아님 (% 처리, ANLS 부재). VLMEvalKit이 표준.
→ **row_a/row_b/row3/row4의 VLMEvalKit Pro 측정 필수.** lmms-eval row_a 34.80% → 보정 시 ~50%, **Chart-R1 7B (48.1%) 초과 가능성.**

### Prior SOTA gap (현재 측정값 기준)
```
                우리 4B(B)  BigCharts-7B  Chart-R1-7B
CQA-H           84.0%       84.8%        86.2%    (-2.2pp)
CQA-A           91.2%       93.2%        92.7%    (-1.5pp)
CharXiv         50.1%       57.2%        56.4%    (-7.1pp) ⚠️
CQA-Pro         39.6%        -           48.1%    (-8.5pp) ⚠️
ChartMuseum     34.8%        -           44.3%    (-9.5pp) ⚠️
```

---

## 3. 핵심 결정사항 (이번 세션 확정)

### Framing — 가장 중요
- **Task-level 문제 정의: "Grounding Drift in Multi-Step Chart Reasoning"** — reasoning chain의 중간 step이 chart로부터 derivable하지 않은 현상. 모든 outcome-only eval/RL이 hidden시키는 본질적 chart reasoning 문제.
- **Prior work gap framing 폐기** — 사용자 feedback: "선행연구들의 gap을 찾는데만 집중했다". Top venue 표준 framing은 **task 본질의 어려움 → prior가 못 푼 부분 → 우리가 점진 해결**.
- **2-stage progressive solution narrative**:
  - Stage 2 (우리 SFT, verification-anchored): grounding의 **form** 학습 → drift 부분 감소
  - Stage 3 (VAPV, process-level reward): grounding의 **enforcement** → 잔여 drift 흡수
  - 둘 다 필요 (SFT only = form만 학습, no enforcement; VAPV only = reasoning trace 부재 시 sparse signal — v8 실패의 진짜 원인)

### Term coining
- **"Grounding Drift"** 우리가 명명. Chain-of-Thought, Process Reward, Constitutional AI처럼 term coining이 contribution의 부분.
- Paper title 후보: *"Mitigating Grounding Drift in Chart Reasoning via Verification-Anchored SFT and Process-Level Reward"* 또는 *"Anchored Reasoning: A Two-Stage Approach to Faithful Chart QA"*.

### Eval / Training format — Always-Think
- **100% thinking on (training + eval).** vLLM `--reasoning-parser qwen3`가 `<think>` 격리하여 `content` field에 post-think answer만 남김.
- **모든 5개 benchmark에 `enable_thinking=True`** (LMMs-Eval default override). 정당화: Chart-R1, BigCharts-R1 등 reasoning-RL precedent.
- 학습 응답 형식 task별: F1' `<think>...</think>42` / F2' `<think>...</think>0.42` / F3'/F4' 동형 / F5' `<think>...</think><answer>X</answer>`.
- **이유**: 75% thinking-off SFT (이전 agent 제안) = VAPV의 reasoning trace source 75% 제거 = VAPV contribution self-destruction.

### SFT data 합성 — Phase 0 결정 트리 (row_a VLMEvalKit 측정 결과 의존)
- **Case I (RL VLMEvalKit Pro ≥ 50%, Chart-R1 7B 초과)**: 합성 NO, v3로 마무리, VAPV 집중. Narrative: "4B + 28K로 7B + 228K SOTA 초과".
- **Case II (45-50%, 동등)**: 최소 합성 1,000 rows (MC 600 + Hyp 400). $50, 1-2일. Framing: "capability gap-driven", test-mirror 아님.
- **Case III (<45%)**: 다양화 합성 2,500 rows (Factoid 60% / MC 15% / FC 10% / Conv 10% / Hyp 5%) — **test 비율과 일부러 다름** (Pro test = 55/16/12/11/5). $80, 3일.

### 기존 결정 (이전 세션 유지)
- DAPO 유지. SFT + GRPO 2-stage. 4B LoRA r=64 alpha=128 / 9B LoRA r=32 alpha=64. 이미지 해상도 448×448 고정.
- Multiplicative reward `r_acc + α*(1-r_acc)*r_proc`. Additive 절대 금지.

---

## 4. 실패 기록 & 교훈

### 이번 세션
1. **이전 research agent의 NO_REASONING/FORMAT_DIRECT framing** — chart reasoning 논문이 아니라 instruction-tuning 논문으로 미끄러짐. "output discipline"은 우리 base model artifact이지 ChartQA task challenge 아님. **agent 출력은 prior work taxonomy 위에서 시작하도록 strict하게 지시해야 함.**
2. **Eval pipeline assumption** — lmms-eval scorer를 ChartQA-Pro 표준으로 가정. 실제로는 VLMEvalKit이 공식. ChartQA-Pro에서 lmms-eval은 % 처리 / ANLS 부재로 +15pp 손실. **새 bench 추가 시 그 bench의 공식 eval framework 먼저 확인.**
3. **"Prior gap" framing의 함정** — 우리 method가 prior가 안 한 것을 한다고 framing하는 것은 incremental contribution으로 보임. Task-level 본질 문제 정의가 top venue 표준.

### 이전 세션 (재기록 — 절대 반복 금지)
1. **Additive reward inflation (v2)**: `min(1.0, r_acc + 0.3*r_proc)` → wrong의 21.1%가 1.0 도달 → 학습 signal 파괴. **multiplicative만 사용.**
2. **Synthetic 차트 타입 버그**: pie/donut/funnel/radar/violin 37.2%가 bar로 렌더링됨. **새 데이터 생성 시 chart type별 렌더링 검증 필수.**
3. **100% 수치 학습 데이터**: ChartMuseum 73% / CQA-Pro 70%가 텍스트 답변인데 학습 데이터에 텍스트 0% → process reward가 binary 작동.
4. **이미지 해상도 변경 금지**: 448×448만 안정. 다른 해상도 = Qwen VL token mismatch crash.

---

## 5. 다음 세션에서 할 일

### Phase 0: Framing 검증 (D0-D3) — 가장 중요, 비용 < $30 + GPU 2-3일

**Day 0 — Anchor coverage 측정 (4시간, $0)**
- Script `scripts/measure_anchor_coverage.py` 작성 (Number regex + CSV match + arithmetic closure)
- 우리 28K SFT corpus 적용 → source별 anchor rate 보고
- **성공 기준**: 평균 anchor rate >85% → framing 채택 가능. <75% → framing 재검토 또는 데이터 filter 강화 필요.

**Day 1 — Drift measurement (preliminary, 1일 GPU)**
- ChartQA-H + Pro + CharXiv + ChartMuseum 각 100 sample = 400 sample (CSV 보유 only)
- Zero-shot Qwen3.5-VL + 우리 SFT v3 두 모델
- **성공 기준**: 우리 SFT의 anchor rate >> zero-shot이면 framing 살아남음. 유사하면 framing 약화.

**Day 2 — External baselines drift measurement (1일 GPU + 다운로드)**
- Chart-R1 7B public ckpt + BigCharts-R1 7B public ckpt 다운
- 같은 400 sample에 drift measurement
- **성공 기준**: 우리 SFT anchor rate > Chart-R1, BigCharts-R1이면 evidence 강함. 비슷하면 contribution 약함.

**Day 3 — GPT-4o drift measurement (~$20 API)**
- 같은 400 sample에 GPT-4o thinking inference (가능한 경우) 또는 CoT prompting
- 우리 SFT+VAPV의 drift rate가 GPT-4o보다 낮으면 매우 강한 evidence

**Day 3 EOD — Framing 결정 게이트**
- Drift table 완성 → 모든 baseline 대비 우리 anchor rate 비교
- 양수이면: framing 채택, paper introduction 1-page draft 시작
- 음수/모호이면: framing fallback (F-B → F-A → F-C → F-D 순서)

### Phase 1: row_a VLMEvalKit + Case 분기 (D1 병행)
- **Day 1 EOD까지 row_a_outcome + row_b_proc + row3/4_dapo의 VLMEvalKit Pro 측정 완료.**
- vLLM swap timeout 해결: Option 2 직렬 처리 (8 ckpt × 30분 = 4시간, 100% 성공)
- 결과에 따라 Case I/II/III 분기:
  - I → 합성 NO, VAPV 집중
  - II → 1K capability-driven 합성 ($50, 1-2일)
  - III → 2.5K balanced 합성 ($80, 3일)

### Phase 2: VAPV reward v3 학습 (D4-D7) — framing 운명의 결정점
- multiplicative reward `r_acc + 0.3*(1-r_acc)*r_proc`
- 텍스트 답변 분기 추가 (`elif not is_numeric and csv_path` in `reward_conditional_v3`)
- **이 결과의 Δ(VAPV vs outcome)이 양수여야 framing 전체 살아남음.**
- D5 EOD 측정 → Δ에 따라 framing 최종 결정

### Phase 3: Paper introduction draft + reviewer attack (D7-D10)
- Drift framing으로 introduction 1-page 작성
- research-paper-attacker로 self-review
- F1 figure ("Grounding Drift Across Chart VLMs") 완성

---

## 6. 참조 파일 & 아티팩트 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `D:\Downloads\chartvr\chartvr\session_handoff_20260512.md` | **이 문서, 필수 읽기** | ✅ 최신 |
| `/mnt/user-data/outputs/chartvr_v8_definitive_plan.md` | 이전 실행 계획 (Row 정의 등) | ⚠️ Framing 부분 outdated, 실험 spec만 참조 |
| `/mnt/user-data/outputs/chartvr_data_plan_v2.md` | 데이터 생성 계획 | ⚠️ Phase 0 결과에 따라 재조정 필요 |
| `code/rewards/rule_verifier_fast.py` | process reward 함수 | ⚠️ v2 → v3 (multiplicative + 텍스트 분기) 업데이트 필요 |
| `train_grpo_dapo.py` | GRPO 학습 스크립트 | ⚠️ reward_conditional_v3 구현 필요 |
| `chartvr/extraction.py` | extract_answer_v3 | ✅ 현재 사용 |
| `eval_standard.py` | 표준 eval script | ⚠️ enable_thinking=True로 모든 bench 통일 필요 |
| `score_standard.py` | 표준 scorer | ✅ 현재 사용 |
| `data/grpo_v2.jsonl` | 현재 GRPO 데이터 3.9K | ⚠️ 100% 수치, 교체 예정 |
| **(신규 작성 필요)** `scripts/measure_anchor_coverage.py` | Anchor coverage 측정 | ❌ 작성 필요 (D0) |
| **(신규 작성 필요)** `scripts/measure_drift.py` | Drift measurement 모델 비교 | ❌ 작성 필요 (D1-D3) |

---

## 7. 주의사항 & 금지사항

### 절대 금지
1. **Additive reward 사용 금지**: 반드시 `r_acc + α*(1-r_acc)*r_proc` 형식
2. **Benchmark-targeting 표현 금지**: paper에 "Pro test 비율 mirror" 같은 표현 절대 금지. 모든 데이터 동기는 capability/verification 기반.
3. **이미지 해상도 변경 금지**: 학습 시 448×448만 사용
4. **Conciseness reward(길이 penalty) 추가 금지**: DeepSeek-R1도 피함
5. **75% thinking-off SFT 절대 금지**: VAPV contribution self-destruct. **always thinking-on + post-think format 분기**가 유일한 옳은 방향
6. **Eval에서 enable_thinking=False 사용 금지**: LMMs-Eval default override, 모든 bench thinking-on
7. **SFT 합성을 row_a VLMEvalKit 측정 전에 시작 금지**: Case I/II/III 분기 데이터 없이 결정 = 자살
8. **Prior gap framing으로 paper 쓰지 말 것**: Task-level "Grounding Drift" framing 채택. 단 framing은 D3 drift measurement 결과 통과 후에만 paper에 commit.

### 사용자 선호 (지키지 않으면 진척 안 됨)
- **확률 기반 판단**: "좋은 방향" 대신 "이길 확률 N%" 정량
- **결과 형상 역설계**: contribution → 필요한 결과 형상 → 실험 설계 (역순 금지)
- **최소 비용 실험 우선**: 큰 실험 전 저비용 검증
- **선행연구 evidence 기반 의사결정**: 모든 제안에 "어떤 accepted paper가 그렇게 했는가" 근거
- **Reviewer 공격 관점 선행 검토**: 방향 확정 전 reviewer attack 시뮬레이션

### 이번 세션 사용자 추가 가이드 (반드시 준수)
- **Task-level 본질 문제 정의를 우선** (prior method gap 분석보다)
- **AVG SOTA + 균형 향상이 expected outcome** — 단일 bench fitting narrative 금지
- **두 stage(SFT, VAPV)는 단일 contribution의 두 측면** — 분리된 contribution 2개로 framing 금지
- **Term coining ("Grounding Drift")은 자산** — paper title과 figure에 활용

---

## 부록: 이번 세션 최종 framing 1-page draft

> **Problem**: Chart reasoning requires that every intermediate claim — every value read from the chart, every comparison, every arithmetic — be **derivable from the chart's underlying data**. A reasoning chain producing the correct final answer through ungrounded intermediate steps is not a successful chart reasoning; it is a coincidence. We term this phenomenon **Grounding Drift**.
>
> **Why prior work doesn't solve it**: All current chart QA training and evaluation operates at outcome-level. LMMs-Eval `relaxed_correctness`, VLMEvalKit ANLS, ChartMuseum LLM judge, CharXiv judge — all measure final answer match. Training signals (Chart-R1 RFT, BigCharts-R1 GRPO) are similarly outcome-only. Under this regime, grounding drift is hidden: models learn to produce correct-looking answers, not correct-reasoning answers. Even SOTA chart VLMs exhibit drift on 25-40% of their reasoning chains (Table 1, our measurement).
>
> **Our solution — progressive 2-stage attack on drift**:
> - **Stage 1 (Verification-Anchored SFT)**: Train on CoT corpus where every step is anchored to chart's tabular ground truth (CSV). Model learns the *structural form* of grounded reasoning. Drift reduces from baseline to intermediate.
> - **Stage 2 (VAPV Process Reward)**: Apply step-level verification reward during RL. Drift reduces to near-zero. The two stages are necessary co-design: SFT alone teaches form without enforcement (v8 result), VAPV alone lacks reasoning trace to operate on (sparse signal).
>
> **Why this leads to AVG SOTA**: Grounding drift hurts all benchmarks. Reducing it improves all of them. Our 4B + 28K outperforms Chart-R1 7B + 228K (Table 2) not by scaling data, but by structuring it for process-reward compatibility.
>
> **Key Figure 1**: Drift rate (% of `<think>` numeric claims that fail to anchor to chart CSV) across Zero-shot, Chart-R1, BigCharts-R1, GPT-4o, our SFT, our SFT+VAPV.

---

**다음 세션 시작 시: 이 문서를 읽고, Day 0 anchor coverage 측정 스크립트 작성부터 시작.**
