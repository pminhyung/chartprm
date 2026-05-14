# v_hq-lite SFT Eval Report (2026-04-11)

## Overview

**Model**: Qwen3.5-4B + LoRA (r=64, alpha=128)
**Training Data**: `sft_hq_lite_clean.jsonl` (43,242 samples, Wait contamination 제거)
**Training**: 676 steps, 1 epoch, DeepSpeed ZeRO-3, 8 GPUs, ~9h
**Eval**: 8× TP=1 vLLM, thinking mode (temp=0.6, top_p=0.95)

## Results Comparison

| Benchmark | v7 Zero-shot | v8 SFT | v8 Outcome | v8 Process | v9 SFT | v9.1 SFT | **v_hq-lite** | vs v8 SFT | vs Best |
|---|---|---|---|---|---|---|---|---|---|
| chartqa_human | 77.84 | 83.68 | **84.48** | 83.76 | 82.32 | 81.76 | **71.20** | -12.48 | -13.28 |
| chartqa_augmented | 86.32 | 88.64 | 88.96 | 88.96 | 91.36 | 87.92 | **94.16** | +5.52 | +2.80 |
| charxiv_reasoning | 45.40 | 44.50 | 46.00 | 45.30 | 42.10 | 39.50 | **40.80** | -3.70 | -5.20 |
| chartqa_pro | 36.09 | 35.83 | 36.40 | 36.14 | 35.11 | 35.99 | **27.98** | -7.85 | -8.42 |
| chartmuseum | 29.90 | 27.60 | 28.30 | 30.40 | 27.30 | 26.40 | **23.80** | -3.80 | -6.60 |
| **5-bench AVG** | **55.11** | **56.05** | **56.83** | **56.91** | **55.64** | **54.31** | **51.59** | **-4.46** | **-5.32** |

*5-bench AVG = simple average of 5 benchmarks*

## Empty Content Analysis (Runaway Indicator)

| Benchmark | v9.1 empty% | v_hq-lite empty% |
|---|---|---|
| chartqa_human | — | 0.0% |
| chartqa_augmented | — | 0.0% |
| charxiv_reasoning | ~runaway | 1.7% |
| chartqa_pro | — | 1.0% |
| chartmuseum | — | 2.4% |

**Runaway thinking 문제 해결**: v9.1에서 CharXiv 44.8% empty content → v_hq-lite에서 1.7%로 완전 해소.

## Key Findings

### 긍정적
1. **Runaway thinking 완전 해결** — empty content 비율이 0-2.4%로 정상 범위
2. **chartqa_augmented 역대 최고 94.16%** — 단순 수치 읽기 능력은 크게 향상 (+5.52pp vs v8 SFT)
3. **학습 안정성 확인** — loss 1.67로 안정 수렴, 학습 과정에서 문제 없음

### 부정적
1. **chartqa_human 71.20%** — 역대 최저, v8 SFT 대비 -12.48pp. 매우 심각한 하락
2. **chartqa_pro 27.98%** — 역대 최저, v8 SFT 대비 -7.85pp
3. **chartmuseum 23.80%** — 역대 최저
4. **charxiv_reasoning 40.80%** — v9.1(39.50%) 대비 +1.30pp이지만, v8(44.50%) 대비 -3.70pp
5. **전체 평균 51.59%** — 모든 이전 버전보다 낮음 (v8 Process 56.91% 대비 -5.32pp)

## Root Cause Analysis

### augmented↑ + human↓ 패턴의 의미
- `chartqa_augmented`는 단순 수치 추출 위주 → SFT 데이터의 QA 포맷으로 잘 학습됨
- `chartqa_human`는 복잡한 해석·추론 질문 → SFT 데이터가 이 유형의 다양성 부족
- **진단**: v_hq-lite 데이터(43K)는 양은 많지만 **질문 다양성 부족** — 생성된 QA가 단순 수치 읽기에 편중

### 전체 하락 원인 가설
1. **데이터 다양성**: v8 SFT는 chartqa_train(18K) + 다양한 소스 기반. v_hq-lite는 자체 생성 QA 중심으로 벤치마크 질문 유형과 분포 불일치
2. **chartqa_train 18K 포함 역효과?**: v_hq-lite에 chartqa_train이 포함되어 있지만, 나머지 25K 자체 생성 데이터가 모델의 일반 추론 능력을 희석시킨 가능성
3. **Non-thinking reasoning format**: SFT 데이터가 `<think>` 없이 reasoning을 학습시켰지만, eval은 thinking mode — 포맷 불일치

## Recommendations

### 즉시 실행
1. **v_hq-lite 데이터 구성 분석** — 43K 샘플의 question type 분포 확인. 단순 수치 읽기 비율 vs 추론 비율
2. **chartqa_human 에러 분석** — 틀린 360개 샘플에서 패턴 확인 (empty answer? wrong reasoning? answer format?)
3. **v8 SFT 데이터와 비교** — v8 SFT의 학습 데이터 구성 대비 v_hq-lite의 차이점 정확 파악

### 전략적 방향
- **Option A**: v8 SFT 데이터(기존) + v_hq-lite 선별 보강 → 혼합 SFT 재학습
- **Option B**: Teacher distilled data (현재 2.7K) 축적 후 reasoning 품질 향상된 데이터로 재학습
- **Option C**: v8 best model(Row B, avg 59.9%)을 baseline으로 GRPO 직행 — SFT 개선 없이
- **Option D**: v_hq-lite에서 augmented 성능이 높은 것을 활용, 데이터 리밸런싱 후 재학습

## Files
- **Results**: `results/v_hq_lite/sft_hq_lite_clean_4b/`
- **Model**: `ckpt/sft_hq_lite_clean_4b_8gpu_merged/`
- **Adapter**: `ckpt/sft_hq_lite_clean_4b_8gpu/`
- **Training data**: `data/sft_hq_lite_clean.jsonl`
- **Training log**: `/tmp/train_logs/sft_hq_lite_clean_v3.log`
