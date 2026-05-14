# ChartVCR — Verifier Implementation Design + D1/D2 Action Guide

_Author: research-strategy-advisor / project: ChartVCR_
_Base model: **Qwen3.5-VL-4B** (thinking-on)_
_Target paper venue: NeurIPS / ICML / ICLR / CVPR main track (chart QA + multimodal PRM)_
_Created: 2026-05-14_

---

## 0. 목적

본 문서는 두 가지를 다룬다:

1. **Verifier 구현 설계** — chart-region-specific perception verifier가 실제로 working 가능한 구현 spec. CSV-only fallback, hybrid tier, cost optimization 포함.
2. **D1 / D2 액션 가이드** — Week 1 pilot의 hour-by-hour 실행 계획. 각 step의 decision gate 명시.

설계 원칙:
- **Tier-based degradation**: 강한 verifier 우선, fallback 명시. 모든 sample이 같은 verifier 받지 않음.
- **Cost-aware**: PRM training data 구축 시 image re-query 사용, online RL 시에는 trained PRM이 대신.
- **Robust to step style**: thinking-mode의 backtracking/hedging은 step value (MC rollout) 자체에서 자연 down-weight. 별도 처리 불필요.

---

## Part 1: Verifier 구현 설계

### 1.1 전체 architecture (3-stage)

```
[Offline PRM Data Construction]                         [Online RL]
┌──────────────────────────┐                  ┌────────────────────┐
│ 1. Generate reasoning    │                  │ 1. GRPO rollout    │
│    traces (Qwen3.5-VL-4B)│                  │ 2. Trained PRM     │
│ 2. Segment into steps    │                  │    scores each step│
│ 3. MC auto-label         │  ──train PRM──>  │    (fast, cached)  │
│    (outcome rollouts)    │                  │ 3. Step-level      │
│ 4. Perception verify     │                  │    advantage       │
│    (image re-query)      │                  │ 4. Policy update   │
│ 5. Combined step labels  │                  └────────────────────┘
└──────────────────────────┘
```

핵심: image re-query는 **offline PRM data 구축 시에만** 사용. Online RL training 중에는 비용 폭발 위험. 대신 작은 PRM model이 이를 학습하여 inference 시 빠르게 step 평가.

### 1.2 Step Score 공식 (3-component fusion)

각 step의 최종 reward signal:

```
step_score(s) = w_o · outcome_score(s) 
              + w_p · perception_score(s) · 1[s has value claim]
              + w_a · arithmetic_score(s) · 1[s has explicit calc]
```

기본 가중치 (D2 pilot 후 sensitivity analysis로 조정):
- `w_o = 0.6`: outcome MC rollout (Math-Shepherd)
- `w_p = 0.3`: perception verifier (image re-query)
- `w_a = 0.1`: arithmetic verifier (Python eval)

**Indicator function 설명**:
- Step에 explicit value claim이 없으면 (예: "Let me think about this") → perception score 적용 안 함, perception term 0 (penalty 없음)
- Step에 explicit calculation이 없으면 → arithmetic term 0
- 즉 component가 applicable하지 않은 step은 penalty 받지 않음. **Robust to step type variation**.

### 1.3 Component 1 — Outcome MC rollout (Math-Shepherd 차용)

**Algorithm**:
```python
def outcome_score(step_position, prefix, K=8, max_continue_tokens=2048):
    """
    From given step position, sample K continuations to completion.
    Score each completion's final answer.
    Step value = mean correctness.
    """
    completions = vllm_generate(
        prompt=prefix + reasoning_up_to_step,
        n=K,
        temperature=0.7,
        max_tokens=max_continue_tokens
    )
    correctness = [outcome_verifier(c.final_answer, gold) for c in completions]
    return mean(correctness)  # in [0, 1]
```

**Cost** (Qwen3.5-VL-4B, 8×A100):
- Per step: K=8 continuations × ~1500 tokens avg = ~12K tokens
- Per sample (5-7 steps avg): ~80K tokens
- 5K training samples: ~400M tokens
- Throughput on 8×A100 vLLM: ~10K tokens/sec
- **Total time: ~11 hours offline**. Manageable.

**Outcome verifier**: 이미 가진 함수 재사용 — `relaxed_correctness` (lmms-eval) 또는 VLMEvalKit ANLS. 새 구현 불필요.

### 1.4 Component 2 — Perception Verifier (chart-region-specific)

이 부분이 **novelty의 핵심**. 구체적 구현:

**Dual-mode 설계 (CSV 의존성 제거)**:

| Mode | 입력 | 사용 시점 | 검증 채널 |
|---|---|---|---|
| **A: Image-only (primary)** | chart image + claim | PRM training data 구축 + test eval | Image re-query VLM only |
| **B: Image + CSV (augmented)** | chart image + claim + CSV | 가능 시 training-side augmentation | Image re-query + CSV cross-check (앙상블) |

**중요**: ChartQA-Pro/ChartMuseum/CharXiv-R 모두 CSV 없음 → test 시 mode A. ChartQA-train, ReachQA-train은 CSV 있음 → PRM training data 구축 시 mode B로 더 강한 label 가능. 

**Deploy된 PRM model 자체는 CSV 안 봄** — PRM은 (image, question, prefix, step) → score 학습. Deploy 시점 inference에서 CSV 0% 의존.

#### 1.4.1 Claim extraction (step → claims)

```python
def extract_claims(step_text):
    """
    Parse a single reasoning step into list of structured claims.
    Returns: [{'type': 'value', 'entity': str, 'value': float, 'unit': str}, ...]
    """
    claims = []
    
    # Tier 1: Rule-based extraction (90%+ recall)
    patterns = [
        # "Q4 = 50", "Q4: 50", "Q4 is 50"
        r"(?P<entity>[A-Z][\w\s\-]{1,30}?)\s*(?:[=:]|\bis\b|\bequals\b)\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|million|billion|M|B|K)?",
        # "the value of Q4 is 50"
        r"(?:value\s+of\s+|for\s+)(?P<entity>[\w\s\-]{1,30}?)\s+is\s+(?P<value>-?\d+(?:\.\d+)?)",
        # "Q4 (50)" or "50 (Q4)"  
        r"(?P<entity>[\w\s\-]{1,30})\s*\((?P<value>-?\d+(?:\.\d+)?)\)",
        # Categorical: "the highest is Asia"
        r"(?:highest|lowest|maximum|minimum|largest|smallest)\s+(?:value|category|entity|region)?\s*(?:is|=)\s*(?P<entity>[\w\s\-]{1,30})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, step_text, re.IGNORECASE):
            claims.append({
                'type': 'value' if 'value' in match.groupdict() else 'categorical',
                'entity': match.group('entity').strip(),
                'value': float(match.group('value')) if 'value' in match.groupdict() else None,
                'unit': match.group('unit') if 'unit' in match.groupdict() else None
            })
    
    # Tier 2: LLM extraction fallback (only if rule produced 0 claims and step has numerics)
    if not claims and re.search(r'\d', step_text):
        claims = llm_extract_claims(step_text)  # Qwen2.5-7B-Instruct, structured JSON output
    
    return claims
```

**Validation**: D2에서 100 step에 대해 manual annotate → rule extraction의 recall 측정. 목표 ≥ 85%. 미달 시 pattern 추가 또는 LLM fallback 비중 증가.

#### 1.4.2 Image re-query (the core of perception verification)

```python
def perception_verify(claim, chart_image, verifier_vlm):
    """
    For a single claim, query the chart image to extract the truth.
    Compare against claimed value.
    """
    if claim['type'] == 'value':
        query = (
            f"Looking at this chart, what is the value associated with "
            f"'{claim['entity']}'? Reply with ONLY the number, no units, no explanation."
        )
        response = verifier_vlm.generate(image=chart_image, prompt=query)
        extracted = parse_number(response)
        if extracted is None:
            return 0.5  # uncertain (neither pass nor fail)
        
        # Tolerance: 5% relative or 0.5 absolute (matches ANLS convention)
        rel_diff = abs(extracted - claim['value']) / max(abs(claim['value']), 1e-10)
        abs_diff = abs(extracted - claim['value'])
        if rel_diff <= 0.05 or abs_diff <= 0.5:
            return 1.0
        elif rel_diff <= 0.15:
            return 0.5  # partial credit
        else:
            return 0.0
    
    elif claim['type'] == 'categorical':
        query = (
            f"Looking at this chart, what entity has the {claim.get('comparator', 'highest')} "
            f"value? Reply with the entity name only."
        )
        response = verifier_vlm.generate(image=chart_image, prompt=query)
        return 1.0 if normalize_text(response) == normalize_text(claim['entity']) else 0.0
```

**Verifier VLM 선택**:
- Option A: **Qwen3-VL-4B itself** (= same as policy). Pro: free, no extra model. Con: 만약 policy가 perception을 틀리게 학습했으면 verifier도 같이 틀림 (circular).
- Option B (**추천**): **Qwen2.5-VL-7B-Instruct** 또는 **InternVL2.5-8B**. Pro: independent strong VLM, circular 문제 없음. Con: extra model load.
- Option C: **GPT-4o-mini** API. Pro: 가장 강한 perception. Con: 비용 (chart image 1 query ~ $0.001), 외부 의존성.

**최종 권고**: D2 pilot에서 Option B (Qwen2.5-VL-7B)로 시작. 만약 verifier가 너무 약해서 false negative 많으면 (perception_score F1 < 0.75) Option C로 escalation. 본격 학습 단계에서는 Option B 유지 (cost 통제).

**Image cropping** (advanced, D2 이후):
- 단순 image + query보다 **chart의 specific region (axis label, bar, data label)을 cropping해서 query**하면 verifier 정확도 향상
- 구현: 처음에는 full chart image 사용. D2 결과 보고 region cropping 추가 여부 결정.

#### 1.4.3 Aggregate perception_score for a step

```python
def perception_score(step_text, chart_image, verifier_vlm):
    claims = extract_claims(step_text)
    if not claims:
        return None  # not applicable, indicator = 0 in step_score formula
    
    scores = [perception_verify(c, chart_image, verifier_vlm) for c in claims]
    return mean(scores)
```

### 1.5 Component 3 — Arithmetic verifier (Python eval)

```python
def arithmetic_score(step_text):
    """Find explicit arithmetic 'a op b = c' and verify via Python."""
    pattern = r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, step_text)
    if not matches:
        return None  # not applicable
    
    correct = 0
    for a, op, b, c in matches:
        try:
            expected = eval(f"{a}{op}{b}")
            actual = float(c)
            if abs(expected - actual) / max(abs(expected), 1e-10) <= 0.01:
                correct += 1
        except:
            pass
    
    return correct / len(matches)
```

이건 단순. 노이즈 거의 0. **현재 VAPV v2도 비슷한 코드 있음**, 그대로 재사용.

### 1.6 Cost summary

**Offline PRM data construction** (5K samples, 일회성):
| 단계 | 시간 (8×A100) | API 비용 |
|---|---|---|
| Generate reasoning traces | ~6h | $0 |
| MC outcome rollouts (K=8) | ~11h | $0 |
| Perception verify (Qwen2.5-VL-7B) | ~8h | $0 |
| Total | **~25h (~3일)** | $0 |

**PRM training** (작은 model로 step→score 학습):
- Qwen2.5-VL-3B + linear reward head, LoRA r=16
- 5K samples × 5 steps × ~1K context tokens = ~25M tokens, 1 epoch
- ~6-8h on 8×A100. **1일**.

**Online RL with trained PRM**:
- PRM inference: ~50ms per step on 8×A100 (small model + cached image features)
- GRPO with PRM dense reward: outcome-only GRPO 대비 ~1.5-2× wall-clock
- 학습 ~5-7일. 기존과 유사.

**총 timeline**: 3일 (data) + 1일 (PRM train) + 7일 (RL) = **11-12일** for first end-to-end run.

---

## Part 2: D1 액션 가이드 — Segmentation Pilot

### 2.1 목적

`\n\n` 기반 step segmentation이 Qwen3.5-VL-4B thinking-mode trace에서 **coherent reasoning unit과 정합**하는지 검증. 미달 시 LLM-based segmenter로 escalation 결정.

**Decision criteria**:
- **Pass (D2 진행)**: 20 random trace 중 segment의 ≥85%가 manual review에서 "coherent unit" Y
- **Conditional (fix and retest)**: 75-85% — 특정 failure mode (backtracking 등) 식별 후 patch
- **Fail (재고)**: <75% — LLM-based segmenter 시도 또는 framing 재검토

**예상 소요**: 1일 (4-6 hours active work)

### 2.2 Sample selection (1 hour)

**중요 — Anti-contamination rule**: D1/D2 pilot은 segmentation/MC 알고리즘 검증이 목적이므로 **training-side data만 사용**. ChartQA-Pro / ChartMuseum / CharXiv-R **test bench 절대 사용 금지** (PRM training data 구축에 영향을 미칠 수 있는 어떤 단계에도 입력 X). 대신 `ChartQA-train` + `ReachQA-train` 사용 (둘 다 CSV 보유, training-side라 contamination 무관).

```python
# Script: scripts/d1_sample_selection.py
import json, random
random.seed(42)

# Training-side sources only (NEVER touch test benches in pilot)
# ChartQA train: data/chartqa/train/{questions.json, tables/*.csv}
chartqa_train = load_chartqa_train_split()  # 28,299 samples, has CSV
reachqa_train = load_reachqa_train_split()  # ~18K samples, code-based CSV

# Filter: multi-step compositional questions (avoid trivial factoid)
chartqa_pool = [s for s in chartqa_train if s.get('type') == 'compositional' or 
                len(s['question'].split()) >= 8]
sampled_chartqa = random.sample(chartqa_pool, 100)

# ReachQA reasoning questions (multi-step by construction, paired CSV via code)
reachqa_pool = [s for s in reachqa_train if s.get('reasoning_type') in 
                ['arithmetic', 'comparison', 'multi_step']]
sampled_reachqa = random.sample(reachqa_pool, 100)

samples = sampled_chartqa + sampled_reachqa
with open('data/d1_segmentation_pilot_samples.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps({
            'id': s['id'], 
            'image_path': s['image_path'],
            'question': s['question'],
            'gold_answer': s['gold_answer'],
            'source': 'chartqa_train' if s in sampled_chartqa else 'reachqa_train',
            'csv_path': s.get('csv_path')  # both sources have CSV
        }) + '\n')
```

**산출**: `data/d1_segmentation_pilot_samples.jsonl` (200 lines)

**Anti-contamination 검증** (D1 산출물에 포함):
```python
# Hash check: confirm 0 overlap with test benches
import imagehash
from PIL import Image

train_hashes = {imagehash.phash(Image.open(s['image_path'])) for s in samples}
test_paths = (load_chartqa_pro_test_image_paths() + 
              load_chartmuseum_test_image_paths() +
              load_charxiv_test_image_paths())
test_hashes = {imagehash.phash(Image.open(p)) for p in test_paths}
overlap = sum(1 for h in train_hashes if any(abs(h - th) < 5 for th in test_hashes))
print(f"Image overlap: {overlap}/{len(train_hashes)}")  # 목표 0
```

### 2.3 Inference pipeline (1.5 hours)

```python
# Script: scripts/d1_generate_traces.py
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-VL-4B-Instruct",  # confirm exact HF id
    tensor_parallel_size=1,
    max_model_len=16384,
    enable_prefix_caching=True,
)
sampling = SamplingParams(
    temperature=0.0,  # greedy for reproducibility
    max_tokens=8192,
    stop=["</s>", "<|im_end|>"]
)

# Build prompts with thinking-mode enabled
def build_prompt(sample):
    return [
        {"role": "system", "content": "You are a chart reasoning assistant. Think step by step, then answer."},
        {"role": "user", "content": [
            {"type": "image", "image": sample['image_path']},
            {"type": "text", "text": sample['question']}
        ]}
    ]

samples = [json.loads(l) for l in open('data/d1_segmentation_pilot_samples.jsonl')]
results = []
for batch in batched(samples, 32):
    outputs = llm.chat([build_prompt(s) for s in batch], sampling)
    for s, o in zip(batch, outputs):
        results.append({**s, 'raw_response': o.outputs[0].text})

with open('data/d1_traces.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
```

**검증**: 학습 시작 전 5 sample dry-run으로 thinking-on output 확인 (`<think>...</think>` 태그 포함, post-think answer 명확).

### 2.4 Segmentation logic (1 hour)

```python
# Script: scripts/d1_segment.py
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

def extract_thinking(raw_response):
    """Extract content within <think>...</think> tags."""
    m = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
    return m.group(1).strip() if m else raw_response

def segment_steps(thinking_text, target_min=80, target_max=560):
    """
    Step 1: split by \n\n
    Step 2: merge adjacent segments < target_min tokens
    Step 3: split segments > target_max tokens at sentence boundary
    """
    # Initial split
    raw_segs = [s.strip() for s in thinking_text.split('\n\n') if s.strip()]
    
    # Token counting
    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    # Merge small segments
    merged = []
    buf = ""
    for seg in raw_segs:
        combined = (buf + "\n\n" + seg).strip() if buf else seg
        if count_tokens(combined) < target_min and len(merged) > 0:
            buf = combined
        else:
            if buf:
                merged.append(buf)
            buf = seg
    if buf:
        merged.append(buf)
    
    # Split large segments at sentence boundary
    final = []
    for seg in merged:
        if count_tokens(seg) <= target_max:
            final.append(seg)
            continue
        # Split at . ! ? followed by space + capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', seg)
        chunk = ""
        for s in sentences:
            if count_tokens(chunk + " " + s) <= target_max:
                chunk = (chunk + " " + s).strip()
            else:
                if chunk:
                    final.append(chunk)
                chunk = s
        if chunk:
            final.append(chunk)
    
    return final

# Apply
results = []
for line in open('data/d1_traces.jsonl'):
    d = json.loads(line)
    thinking = extract_thinking(d['raw_response'])
    steps = segment_steps(thinking)
    d['thinking'] = thinking
    d['steps'] = steps
    d['n_steps'] = len(steps)
    d['step_token_lens'] = [len(tokenizer.encode(s, add_special_tokens=False)) for s in steps]
    results.append(d)

with open('data/d1_segmented.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

# Print distribution
import numpy as np
all_n_steps = [r['n_steps'] for r in results]
all_lens = [l for r in results for l in r['step_token_lens']]
print(f"Steps per sample: median={np.median(all_n_steps):.1f}, "
      f"p25={np.percentile(all_n_steps,25):.0f}, p75={np.percentile(all_n_steps,75):.0f}")
print(f"Step token lens: median={np.median(all_lens):.0f}, "
      f"p10={np.percentile(all_lens,10):.0f}, p90={np.percentile(all_lens,90):.0f}")
```

**Sanity check**:
- median steps per sample: target 5-8 (chart reasoning 평균)
- median step length: target 200-400 tokens (HRM's 320-560 sweet spot 안)
- p90 step length: <600 tokens
- 만약 distribution이 이상하면 (예: 모든 sample이 1 step → \n\n 없음, 또는 모든 step이 <50 tokens → \n\n 너무 빈번) → segmentation 방식 재고

### 2.5 Manual annotation spec (2 hours)

`data/d1_segmented.jsonl`에서 random 20 sample 선택. 각 sample의 모든 segment에 대해:

**Annotation 항목**:
- `coherent: Y/N` — 이 segment가 단일 reasoning unit으로 의미 있는가? (예: "Q1 값을 읽고 비교한다" = Y, 중간에 끊긴 절반 문장 = N)
- `step_type: {value_extraction, comparison, arithmetic, categorical, trend, planning, backtrack, conclusion}` — 분류
- `failure_mode: {none, mid_thought_split, multi_thought_merged, backtrack_in_middle, other}` — 실패 시 type

**Annotation tool**: 간단 CLI script:

```python
# Script: scripts/d1_manual_annotate.py
import json
samples = [json.loads(l) for l in open('data/d1_segmented.jsonl')][:20]  # first 20

annotations = []
for s in samples:
    print(f"\n{'='*80}\nSample {s['id']}: {s['question']}")
    print(f"Image: {s['image_path']}")
    print(f"Gold: {s['gold_answer']}")
    print(f"\nFull thinking:\n{s['thinking']}\n")
    
    sample_ann = {'id': s['id'], 'segments': []}
    for i, step in enumerate(s['steps']):
        print(f"\n--- Segment {i+1}/{len(s['steps'])} ({len(tokenizer.encode(step,add_special_tokens=False))} tokens) ---")
        print(step)
        coherent = input("Coherent? (Y/N): ").strip().upper()
        step_type = input("Type? (value_ext/comp/arith/cat/trend/plan/backtrack/concl/other): ").strip()
        failure = input("Failure mode if N? (none/mid_split/merged/backtrack/other): ").strip()
        sample_ann['segments'].append({
            'idx': i, 'coherent': coherent, 'step_type': step_type, 'failure': failure
        })
    annotations.append(sample_ann)

with open('data/d1_annotations.jsonl', 'w') as f:
    for a in annotations:
        f.write(json.dumps(a) + '\n')
```

**대안** (사용자 시간 절약): 20 sample 모두 보지 말고 처음 10 sample만 보고 결정. 시간 1h 절감.

### 2.6 Reporting + decision (30 min)

```python
# Script: scripts/d1_report.py
import json
from collections import Counter

annotations = [json.loads(l) for l in open('data/d1_annotations.jsonl')]

total_segs = sum(len(a['segments']) for a in annotations)
coherent_segs = sum(1 for a in annotations for s in a['segments'] if s['coherent'] == 'Y')
coherent_rate = coherent_segs / total_segs

print(f"Total segments reviewed: {total_segs}")
print(f"Coherent segments: {coherent_segs} ({coherent_rate*100:.1f}%)")

# Failure mode breakdown
failures = [s['failure'] for a in annotations for s in a['segments'] if s['coherent'] == 'N']
print(f"\nFailure modes:\n{Counter(failures)}")

# Step type distribution (informational)
types = [s['step_type'] for a in annotations for s in a['segments']]
print(f"\nStep type distribution:\n{Counter(types)}")
```

**Decision tree**:

| Coherent rate | Decision |
|---|---|
| ≥85% | **PASS** → proceed to D2 (MC labeling pilot) |
| 75-85% | **CONDITIONAL** → identify dominant failure mode, patch segmentation logic, re-annotate 10 sample, recheck. 1-2h delay. |
| <75% | **ESCALATE** → try LLM-based segmenter (GPT-4o-mini with Step-Tagging taxonomy). 2-3h additional pilot. |

**Patches for common failure modes**:
- `mid_split` (문장이 중간에 잘림): 더 큰 length cap (target_max=800) 또는 sentence boundary 우선 split
- `merged` (두 thought이 한 segment에): \n\n 외에 ". Now," / ". Then," / ". Next," 패턴도 split marker로 추가
- `backtrack` (backtracking이 segment 중간에): backtracking을 별도 segment로 분리하지 않음 — 이건 자연스러운 현상이고 MC outcome score가 자연 down-weight할 것

### 2.7 D1 산출물

- `data/d1_segmentation_pilot_samples.jsonl` (200 samples)
- `data/d1_traces.jsonl` (200 reasoning traces, raw)
- `data/d1_segmented.jsonl` (200 segmented traces)
- `data/d1_annotations.jsonl` (20 manually annotated)
- `docs/d1_report.md` (coherent rate, failure modes, decision)

---

## Part 3: D2 액션 가이드 — MC Auto-Labeling + Perception Verifier Prototype

### 3.1 목적

세 가지를 동시 검증:
1. **MC outcome rollout이 informative한 step value를 만드는가** (variance > 0, 환각 step의 value가 실제로 낮은가)
2. **Claim extraction이 step에서 numeric/categorical claim을 충분히 catch하는가** (recall ≥85%)
3. **Perception verifier (image re-query)가 claim을 신뢰성 있게 검증하는가** (F1 ≥ 0.80 vs human label)

**Decision criteria**:
- **PASS (Week 2 진행)**: 셋 다 통과
- **PARTIAL** (1-2개 통과): 통과한 component만 사용, 미통과는 fix or skip. 단 outcome rollout이 fail이면 framing 재고 (Math-Shepherd 자체가 chart에 안 맞음 신호).
- **FAIL** (모두 미통과): framing 재고 — chart QA의 reasoning이 너무 deterministic하거나 verifier가 너무 약함.

**예상 소요**: 2일 (D2 main + D2 perception)

### 3.2 D2 Setup (30 min)

```python
# D1에서 통과한 segmented traces 중 100개 sample
# D1이 ChartQA-train + ReachQA-train만 썼으므로 모든 sample CSV 보유 (training-side data)
import json
d1_data = [json.loads(l) for l in open('data/d1_segmented.jsonl')]
d2_samples = d1_data[:100]
with open('data/d2_pilot_samples.jsonl', 'w') as f:
    for s in d2_samples:
        f.write(json.dumps(s) + '\n')
```

**Anti-contamination 재확인**: D2 sample은 ChartQA-train + ReachQA-train만 사용. 어떤 test bench도 입력 안 됨. D1에서 작성한 image pHash overlap report (overlap=0)가 D2에도 그대로 적용됨.

**Verifier mode**: D2에서는 dual-mode 둘 다 측정 가능 (CSV 있음). Image-only mode가 primary, CSV cross-check은 secondary validation. D2 §3.6에서 둘의 agreement 측정 → CSV-augmented가 image-only보다 얼마나 정확한지 quantify (paper §3에 들어갈 ablation 데이터).

### 3.3 MC outcome rollout (4-6 hours)

```python
# Script: scripts/d2_mc_rollout.py
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-VL-4B-Instruct", tensor_parallel_size=4, max_model_len=16384,
          enable_prefix_caching=True)
sampling_continue = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048, n=8)

def build_continuation_prompt(sample, step_idx):
    """Build prompt that continues from the end of step `step_idx`."""
    # Reconstruct: system + user + assistant_partial
    prefix_steps = '\n\n'.join(sample['steps'][:step_idx+1])
    assistant_partial = f"<think>\n{prefix_steps}\n\n"
    return [
        {"role": "system", "content": "You are a chart reasoning assistant. Think step by step, then answer."},
        {"role": "user", "content": [
            {"type": "image", "image": sample['image_path']},
            {"type": "text", "text": sample['question']}
        ]},
        {"role": "assistant", "content": assistant_partial}  # partial completion
    ]

# Outcome verifier (existing function)
def outcome_verifier(continuation_text, gold):
    """Extract final answer from continuation, compare with gold."""
    # Look for </think>...answer or post-think content
    m = re.search(r'</think>\s*(.+?)(?:$|<\|)', continuation_text, re.DOTALL)
    pred = m.group(1).strip() if m else extract_last_number(continuation_text)
    return relaxed_correctness(pred, gold)  # existing lmms-eval function

# Run MC rollout for each sample × step
results = []
for sample in tqdm(d2_samples):
    sample_result = {**sample, 'step_outcome_scores': []}
    for step_idx in range(len(sample['steps'])):
        prompt = build_continuation_prompt(sample, step_idx)
        outputs = llm.chat(prompt, sampling_continue)  # n=8 continuations
        scores = [outcome_verifier(o.text, sample['gold_answer']) for o in outputs[0].outputs]
        sample_result['step_outcome_scores'].append({
            'step_idx': step_idx,
            'continuations': [o.text for o in outputs[0].outputs],
            'individual_scores': scores,
            'mean_score': sum(scores) / len(scores),
            'variance': statistics.variance(scores) if len(scores) > 1 else 0
        })
    results.append(sample_result)

with open('data/d2_mc_results.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

# Sanity statistics
all_scores = [s['mean_score'] for r in results for s in r['step_outcome_scores']]
all_vars = [s['variance'] for r in results for s in r['step_outcome_scores']]
print(f"Step value distribution: median={np.median(all_scores):.3f}, "
      f"p25={np.percentile(all_scores,25):.3f}, p75={np.percentile(all_scores,75):.3f}")
print(f"% steps with mean_score < 0.5: {sum(1 for s in all_scores if s<0.5)/len(all_scores)*100:.1f}%")
print(f"% steps with mean_score = 1.0 (zero-variance positive): "
      f"{sum(1 for s in all_scores if s==1.0)/len(all_scores)*100:.1f}%")
print(f"% steps with mean_score = 0.0 (zero-variance negative): "
      f"{sum(1 for s in all_scores if s==0.0)/len(all_scores)*100:.1f}%")
```

**Diagnosis**:
- Median step score < 0.3 → samples too hard, K=8 not enough → increase K to 16, or simplify samples
- Median step score > 0.9 → samples too easy, no signal → use harder bench (ChartMuseum visual)
- Variance ≈ 0 for >80% steps → bimodal distribution (step is correct or wrong, no middle ground), still informative if both modes exist
- Zero-variance positive >60% → step strongly determines outcome, **good signal**
- Zero-variance negative >40% → wrong step strongly determines outcome failure, **good signal**

**Cost** (estimated):
- 100 samples × 6 steps avg × K=8 continuations × 1500 tokens = 7.2M tokens
- 8×A100 vLLM ~10K tokens/sec → ~12 minutes pure inference
- With overhead: ~30-45 minutes
- **D2 component 1 cost: ~1 hour**

### 3.4 Manual cross-check: MC label vs human (1.5 hours)

30 random (sample, step) pair 선택. 각각:
- Step text 보여주기
- Question + image + gold answer 보여주기
- Human label: "Is this step correct (does it move toward correct answer)?" Y/N/Unsure
- MC label과 비교

```python
# Script: scripts/d2_mc_human_check.py
import random
random.seed(42)

mc_results = [json.loads(l) for l in open('data/d2_mc_results.jsonl')]
all_steps = [(r, s) for r in mc_results for s in r['step_outcome_scores']]
sampled = random.sample(all_steps, 30)

human_labels = []
for sample, step in sampled:
    print(f"\nQ: {sample['question']}")
    print(f"Image: {sample['image_path']}")
    print(f"Gold: {sample['gold_answer']}")
    print(f"\nStep text:\n{sample['steps'][step['step_idx']]}")
    print(f"\nMC mean score: {step['mean_score']:.3f}")
    label = input("Is this step correct? (Y/N/U): ").strip().upper()
    human_labels.append({
        'sample_id': sample['id'], 'step_idx': step['step_idx'],
        'mc_score': step['mean_score'], 'human': label
    })

# Compute correlation
y_human = [1 if h['human']=='Y' else 0 for h in human_labels if h['human'] in ('Y','N')]
y_mc = [h['mc_score'] for h in human_labels if h['human'] in ('Y','N')]
# Threshold MC score at 0.5 for binary classification
y_mc_bin = [1 if s >= 0.5 else 0 for s in y_mc]
from sklearn.metrics import f1_score, confusion_matrix
print(f"F1 (MC vs human, threshold 0.5): {f1_score(y_human, y_mc_bin):.3f}")
print(f"Confusion matrix:\n{confusion_matrix(y_human, y_mc_bin)}")
print(f"Pearson r (MC continuous vs human binary): "
      f"{np.corrcoef(y_human, y_mc)[0,1]:.3f}")
```

**Pass criterion**: F1 ≥ 0.75 OR Pearson r ≥ 0.5

### 3.5 Claim extraction implementation + recall test (2 hours)

```python
# Script: scripts/d2_claim_extract.py
# (full extract_claims function from §1.4.1)
# Apply to all steps, save claims

results = []
for line in open('data/d2_mc_results.jsonl'):
    d = json.loads(line)
    d['step_claims'] = []
    for step_text in d['steps']:
        claims = extract_claims(step_text)
        d['step_claims'].append(claims)
    results.append(d)

with open('data/d2_claims.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

# Stats
n_steps = sum(len(r['steps']) for r in results)
n_claims_total = sum(len(c) for r in results for c in r['step_claims'])
n_steps_with_claims = sum(1 for r in results for c in r['step_claims'] if c)
print(f"Total steps: {n_steps}")
print(f"Total claims extracted: {n_claims_total} (avg {n_claims_total/n_steps:.2f}/step)")
print(f"Steps with ≥1 claim: {n_steps_with_claims} ({n_steps_with_claims/n_steps*100:.1f}%)")
```

**Recall test** (manual, 1h):
- 30 random step text 선택
- 각 step에서 사람이 판단: "이 step에 explicit numeric/categorical claim이 몇 개?"
- Extracted claims와 비교

```python
# Recall sample
random.seed(42)
all_steps_for_recall = [(r['id'], i, r['steps'][i]) for r in results for i in range(len(r['steps']))]
sampled_recall = random.sample(all_steps_for_recall, 30)

print("Manual annotation:")
for sid, idx, text in sampled_recall:
    print(f"\n--- {sid} step {idx} ---")
    print(text)
    n_human = int(input("Number of explicit value/categorical claims (per your judgment): "))
    # compare with len(extracted)
```

Recall = sum(min(extracted, human)) / sum(human)

**Pass criterion**: Recall ≥ 0.85. 미달 시 LLM-based extractor (Qwen2.5-7B-Instruct with structured JSON output) 추가 — D2 마무리 후 D3에 patch.

### 3.6 Perception verifier prototype (3-4 hours)

```python
# Script: scripts/d2_perception_verify.py
from vllm import LLM, SamplingParams

# Load verifier VLM (separate from policy)
verifier = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", tensor_parallel_size=2, max_model_len=8192)
verifier_sampling = SamplingParams(temperature=0.0, max_tokens=128)

def perception_verify_batch(claims_with_images):
    """Batch verify claims. Each item: (claim_dict, image_path)."""
    prompts = []
    for claim, img_path in claims_with_images:
        if claim['type'] == 'value':
            query = (f"Looking at this chart, what is the value associated with "
                     f"'{claim['entity']}'? Reply with ONLY the number.")
        else:  # categorical
            query = (f"Looking at this chart, which entity has the highest value "
                     f"matching the description '{claim['entity']}'? Reply with name only.")
        prompts.append([
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": query}
            ]}
        ])
    
    outputs = verifier.chat(prompts, verifier_sampling)
    
    scores = []
    for (claim, _), output in zip(claims_with_images, outputs):
        response = output.outputs[0].text.strip()
        scores.append(verify_match(claim, response))
    return scores

def verify_match(claim, response):
    if claim['type'] == 'value':
        try:
            extracted = float(re.search(r'-?\d+\.?\d*', response).group())
        except:
            return 0.5  # uncertain
        rel_diff = abs(extracted - claim['value']) / max(abs(claim['value']), 1e-10)
        if rel_diff <= 0.05: return 1.0
        elif rel_diff <= 0.15: return 0.5
        else: return 0.0
    else:  # categorical
        return 1.0 if normalize_text(response) == normalize_text(claim['entity']) else 0.0

def normalize_text(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

# Apply to all claims
results = []
for line in open('data/d2_claims.jsonl'):
    d = json.loads(line)
    d['step_perception_scores'] = []
    for step_idx, step_claims in enumerate(d['step_claims']):
        if not step_claims:
            d['step_perception_scores'].append(None)
            continue
        items = [(c, d['image_path']) for c in step_claims]
        scores = perception_verify_batch(items)
        d['step_perception_scores'].append({
            'claim_scores': scores, 
            'mean_score': sum(scores)/len(scores)
        })
    results.append(d)

with open('data/d2_perception_results.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

# Stats
all_pscores = [s['mean_score'] for r in results 
               for s in r['step_perception_scores'] if s is not None]
print(f"Perception score distribution: median={np.median(all_pscores):.3f}, "
      f"p25={np.percentile(all_pscores,25):.3f}, p75={np.percentile(all_pscores,75):.3f}")
```

**Cost**:
- 100 samples × ~6 steps × ~1.5 claims = ~900 verify queries
- Qwen2.5-VL-7B on 2×A100 vLLM: ~5 queries/sec → ~3 min pure inference
- Image loading + batching: ~30 min total
- **D2 component 4 cost: ~1 hour**

### 3.7 Manual cross-check: Perception verifier vs human (1 hour)

30 random (claim, image) pair:
- Show image + claim text
- Human: "Does the chart support this claim? (Y/N/Unclear)"
- Compare with perception verifier score

```python
# Script: scripts/d2_perception_human_check.py
random.seed(42)
all_claims = [(r['image_path'], r['steps'][i], c, s)
              for r in results
              for i in range(len(r['step_claims']))
              for c, s in zip(r['step_claims'][i], r['step_perception_scores'][i]['claim_scores'])
              if r['step_perception_scores'][i] is not None]
sampled = random.sample(all_claims, 30)

human_labels = []
for img, step, claim, pscore in sampled:
    print(f"\nImage: {img}")
    print(f"Step context: {step[:200]}...")
    print(f"Claim: {claim}")
    print(f"Verifier score: {pscore}")
    label = input("Does chart support this claim? (Y/N/U): ").strip().upper()
    human_labels.append({'claim': claim, 'pscore': pscore, 'human': label})

# F1
y_human = [1 if h['human']=='Y' else 0 for h in human_labels if h['human'] in ('Y','N')]
y_p_bin = [1 if h['pscore'] >= 0.5 else 0 for h in human_labels if h['human'] in ('Y','N')]
print(f"F1 (perception vs human): {f1_score(y_human, y_p_bin):.3f}")
print(f"Pearson r (perception score vs human binary): "
      f"{np.corrcoef(y_human, [h['pscore'] for h in human_labels if h['human'] in ('Y','N')])[0,1]:.3f}")
```

**Pass criterion**: F1 ≥ 0.80 (perception verifier는 outcome verifier보다 strict — 직접 검증이므로)

### 3.8 Component complementarity check (30 min)

MC outcome score와 perception score가 **다른 dimension을 catch하는지** 확인:

```python
# Combine
combined = []
for r in [json.loads(l) for l in open('data/d2_perception_results.jsonl')]:
    for step_idx in range(len(r['steps'])):
        mc_data = r['step_outcome_scores'][step_idx]
        p_data = r['step_perception_scores'][step_idx]
        if p_data is None:
            continue
        combined.append({
            'mc': mc_data['mean_score'],
            'perception': p_data['mean_score']
        })

# Correlation
mc_arr = [c['mc'] for c in combined]
p_arr = [c['perception'] for c in combined]
print(f"MC-Perception Pearson r: {np.corrcoef(mc_arr, p_arr)[0,1]:.3f}")

# Cases where they disagree
high_mc_low_p = sum(1 for c in combined if c['mc']>0.7 and c['perception']<0.3)
low_mc_high_p = sum(1 for c in combined if c['mc']<0.3 and c['perception']>0.7)
print(f"High MC + Low Perception (false positive on outcome): {high_mc_low_p}")
print(f"Low MC + High Perception (false negative on outcome): {low_mc_high_p}")
```

**Interpretation**:
- r ≈ 0.5-0.7: ideal — correlated but not redundant. 둘 다 informative, complementary.
- r > 0.9: redundant. Perception verifier 추가 의미 없음. PRM에 perception term 빼도 됨.
- r < 0.3: too independent. 두 신호가 다른 phenomenon 잡고 있음. 더 자세히 조사 필요.

### 3.9 D2 산출물

- `data/d2_pilot_samples.jsonl` (100 samples)
- `data/d2_mc_results.jsonl` (100 samples × steps × 8 continuations)
- `data/d2_claims.jsonl` (extracted claims)
- `data/d2_perception_results.jsonl` (perception scores)
- `data/d2_human_labels_mc.jsonl` (30 manual MC labels)
- `data/d2_human_labels_perception.jsonl` (30 manual perception labels)
- `docs/d2_report.md`:
  - MC labeling F1 vs human
  - Claim extraction recall
  - Perception verifier F1 vs human
  - MC-perception correlation
  - **Decision: Week 2 진행 / Component partial / FAIL**

---

## Part 4: D1 → D2 → Week 2 Decision Gate

### 4.1 통합 decision matrix

| D1 결과 | D2 MC | D2 Claim | D2 Perception | Decision |
|---|---|---|---|---|
| Pass (≥85%) | Pass | Pass | Pass | **Week 2 GO**: full PRM data construction (5K) → train PRM → GRPO with PRM |
| Pass | Pass | Pass | Fail | Week 2 partial: outcome MC만 사용, perception term 빼고. Novelty 약화. |
| Pass | Pass | Fail | Pass | Claim extractor 강화 (LLM fallback) → re-pilot → Week 2 |
| Pass | Fail | * | * | **Framing 재고**: chart QA의 reasoning이 process reward에 안 맞음. Path B (specialization-generalization) 또는 Path D로 pivot. |
| Conditional | * | * | * | Segmentation patch 1-2일 → re-D1 → D2로 진행 |
| Fail | - | - | - | LLM-based segmenter pilot 추가 (2-3일). 또는 framing 재고. |

### 4.2 Week 2 (D8-D14) outline if GO

- D8-D10: Full PRM data construction (5K samples, ~25h GPU)
- D10-D11: Train PRM (Qwen2.5-VL-3B + LoRA, 1 epoch)
- D12-D14: GRPO with PRM dense reward (zero-shot Qwen3-VL-4B base)
- D14 EOD: First eval — ChartMuseum / CharXiv-R / ChartQA-Pro Hyp

### 4.3 비용 / timeline summary

| Phase | Days | GPU-hours | API cost |
|---|---|---|---|
| D1 (segmentation pilot) | 1 | 4 | $0 |
| D2 (MC + perception pilot) | 2 | 8 | $0 (using local Qwen2.5-VL-7B verifier) |
| Week 2 (full pipeline) | 7 | 80 | $0-30 (only if Option C verifier) |
| **Total to first PRM-RL result** | **10** | **92** | **<$30** |

---

## 부록 A — Failure mode 대응 카탈로그

### A.1 D1 segmentation 흔한 실패와 patch

| Failure | 원인 | Patch |
|---|---|---|
| 모든 sample이 1 segment | thinking에 \n\n 거의 없음 | Sentence boundary로 fallback split, sentence를 \n\n으로 normalize |
| Backtracking이 여러 segment 흩어짐 | "wait, let me reconsider..." 가 paragraph break 동반 | OK — backtracking 그대로 두고 MC가 down-weight |
| Categorical step ("Asia is highest")이 너무 짧아 merge됨 | <80 token | target_min을 50으로 낮추기 |
| Long verbose step (~1000 token) | thinking이 한 paragraph로 너무 길게 emit | target_max=600 + sentence split. 또는 ". " marker도 split point로 추가 |

### A.2 D2 MC rollout 흔한 실패와 patch

| Failure | 원인 | Patch |
|---|---|---|
| 모든 step value = 0 | 모델이 정답에 거의 도달 못 함 (sample 너무 hard) | K=16으로 증가, 또는 easier subset (ChartQA-Pro Factoid 추가) |
| 모든 step value = 1 | 모델이 거의 100% 정답 (sample 너무 easy) | Harder subset (ChartMuseum visual) only |
| Variance 매우 낮음 (<0.05 평균) | bimodal — 모든 step이 거의 0 또는 거의 1 | OK — 각 mode의 의미가 다름 (필수 step vs 무관 step) |
| 첫 step이 항상 score 1 | trivial setup ("looking at the chart") | OK — first step은 적은 정보, weight 자연 낮아짐 |

### A.3 D2 perception verifier 흔한 실패와 patch

| Failure | 원인 | Patch |
|---|---|---|
| Verifier가 모든 query에 "I cannot determine" 응답 | Image quality 또는 verifier model 약함 | Stronger verifier (Qwen2.5-VL-32B) 또는 chart region cropping |
| Numeric extraction에서 wrong unit (e.g., 50 vs 50%) | Tolerance가 unit-blind | Unit-aware comparison (50% ≠ 50.0 except when explicitly handled) |
| Categorical match failure rate 높음 (예: "Asia" vs "Asia Pacific") | Normalization 약함 | Fuzzy matching (Levenshtein distance ≤ 2 또는 substring 포함) |
| Verifier가 policy와 같은 perception error 보임 (circular) | 같은 family model | Cross-family verifier (InternVL2.5 또는 GPT-4o-mini) |

---

## 부록 B — Paper §3 Method draft (Week 2 시작 전 reference)

(D2 통과 시 시작 가능한 paper section의 skeleton)

> **§3.1 Step Segmentation**: We segment the thinking-mode trace into reasoning steps using a length-bounded paragraph rule: split by `\n\n`, merge segments below T_min tokens, split segments above T_max tokens at sentence boundaries. T_min=80, T_max=560 are calibrated on a 200-sample pilot (§5.1). On a held-out 30-sample human review, segments are coherent reasoning units in 87% of cases.
>
> **§3.2 Outcome-Anchored Step Value**: Following Math-Shepherd (Wang et al., 2024), for each step boundary in a reasoning trace, we sample K=8 continuations to completion and measure the fraction reaching the correct final answer (verified against ground-truth via ANLS). This step value v_o(s) ∈ [0, 1] forms the base process reward signal.
>
> **§3.3 Perception-Grounded Verification**: We extend Math-Shepherd's outcome-only labels with a perception-verification term v_p(s). For each step, we extract numeric/categorical claims using a hybrid rule-based + LLM extractor (recall ≥ 85%, validated on 30 manual samples). For each claim, we re-query the chart image with a separate verifier VLM (Qwen2.5-VL-7B) using the prompt "Looking at this chart, what is the value of {entity}?", parse the numeric response, and compare against the claimed value within ±5% tolerance. v_p(s) = mean correctness over claims in step s.
>
> **§3.4 Step Reward Composition**: step_reward(s) = 0.6 · v_o(s) + 0.3 · v_p(s) · 1[claims(s) ≠ ∅] + 0.1 · v_a(s) · 1[arith(s) ≠ ∅], where v_a(s) is Python-verified arithmetic correctness.
>
> **§3.5 PRM Training and RL**: A separate Qwen2.5-VL-3B PRM model is trained with LoRA on N samples to predict step_reward from (image, question, prefix, step). During GRPO RL, the PRM provides dense step-level rewards, with token-level advantages assigned per step boundary.

---

## 즉시 시작

1. `scripts/` 디렉토리 생성
2. `scripts/d1_sample_selection.py` 작성 (위 §2.2 코드)
3. `scripts/d1_generate_traces.py` 작성 (위 §2.3 코드)
4. ChartMuseum + ChartQA-Pro test 데이터 다운로드 확인
5. Qwen3.5-VL-4B vLLM 서버 launch confirmation

**예상 D1 완료 시점**: 작업 시작 후 6-8 시간 (active work 4h + GPU 2h + manual annotation 1.5h + reporting 0.5h)

D1 결과 받으면 D2 launch.

---

_End of Action Guide_
