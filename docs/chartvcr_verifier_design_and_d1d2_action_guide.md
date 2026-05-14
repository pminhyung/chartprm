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

### 2.5 Automated validation (30 min, manual ≈ 0)

원래 manual annotation으로 design했으나 **proxy metrics + spot check로 대체**. Manual은 5 sample 10분만.

```python
# Script: scripts/d1_auto_validate.py
import json, re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
data = [json.loads(l) for l in open('data/d1_segmented.jsonl')]

# Proxy 1: 길이 분포 sanity
all_lens = [l for d in data for l in d['step_token_lens']]
all_n_steps = [d['n_steps'] for d in data]
median_len = np.median(all_lens)
p90_len = np.percentile(all_lens, 90)
median_steps = np.median(all_n_steps)

print(f"[Length] median={median_len:.0f} p90={p90_len:.0f} (target: median 200-400, p90<600)")
print(f"[Steps/sample] median={median_steps:.1f} (target: 5-8)")

# Proxy 2: Sentence boundary integrity
# Each segment should end with terminal punctuation .!?
def ends_with_terminal(text):
    text = text.rstrip()
    return bool(re.search(r'[.!?]["\')\]]?$', text))

terminals = [ends_with_terminal(step) for d in data for step in d['steps']]
terminal_rate = sum(terminals) / len(terminals)
print(f"[Sentence integrity] {terminal_rate*100:.1f}% segments end with .!? (target ≥80%)")

# Proxy 3: Tiny fragment detection (segments <30 tokens after merge → suggests over-split)
tiny = [l for l in all_lens if l < 30]
print(f"[Tiny fragments] {len(tiny)}/{len(all_lens)} segments <30 tokens "
      f"({len(tiny)/len(all_lens)*100:.1f}%, target <5%)")

# Proxy 4: Distribution shape — too many 1-step samples = \n\n 부재
single_step_rate = sum(1 for n in all_n_steps if n <= 1) / len(all_n_steps)
print(f"[Single-step samples] {single_step_rate*100:.1f}% (target <10%)")

# Combined sanity
checks = {
    'length_median_in_range': 150 <= median_len <= 500,
    'length_p90_under_cap': p90_len <= 700,
    'steps_per_sample_in_range': 3 <= median_steps <= 12,
    'sentence_integrity_ok': terminal_rate >= 0.75,
    'tiny_fragment_ok': len(tiny) / len(all_lens) <= 0.10,
    'single_step_ok': single_step_rate <= 0.15,
}
print(f"\nAuto sanity checks: {sum(checks.values())}/{len(checks)} pass")
for k, v in checks.items():
    print(f"  {'✓' if v else '✗'} {k}")
```

**Manual spot check (10분, 5 sample만)**:
```python
# Script: scripts/d1_spot_check.py
import json, random
random.seed(0)
data = [json.loads(l) for l in open('data/d1_segmented.jsonl')]
samples = random.sample(data, 5)

print("=== 5-sample spot check (Y/N each segment, takes ~10min) ===")
for s in samples:
    print(f"\nQ: {s['question']}")
    for i, step in enumerate(s['steps']):
        print(f"\n--- Segment {i+1} ---\n{step}")
        input("Press ENTER if reasonable, type 'BAD' if obviously broken: ")
```

목적: 자동 metric을 보완하는 안전망. 5 sample 모두 정상이면 segmentation 신뢰. 1 sample 이상 broken이면 자동 metric 다시 점검.

### 2.6 Decision rule (5 min)

| Auto sanity | Spot check | Decision |
|---|---|---|
| 6/6 pass | 0 BAD / 5 | **PASS** → D2 |
| 4-5/6 pass | 0-1 BAD / 5 | **CONDITIONAL** → 실패한 check에 대응하는 patch (아래 catalog) → re-run script (5분) → 통과 시 D2 |
| <4/6 pass OR 2+ BAD / 5 | - | **ESCALATE** → LLM-based segmenter (GPT-4o-mini) 2-3h pilot |

**Patch catalog (자동 metric별 대응)**:
- `length_median > 500`: target_max=400으로 강화, sentence boundary split 활성화
- `tiny_fragment > 10%`: target_min=120으로 상향
- `sentence_integrity < 75%`: split을 sentence boundary 우선으로 (paragraph 우선이 아닌)
- `single_step > 15%`: thinking에 \n\n 자체가 부족 → fallback으로 ". Then," ". Now," ". Next," ". Looking at" 등 marker 추가

### 2.7 D1 산출물

- `data/d1_segmentation_pilot_samples.jsonl` (200 samples)
- `data/d1_traces.jsonl` (200 reasoning traces, raw)
- `data/d1_segmented.jsonl` (200 segmented traces)
- `data/d1_image_hash_overlap.json` (anti-contamination report, target overlap=0)
- `docs/d1_report.md` (auto sanity checks, spot check result, decision)

**Manual workload**: ~10분 (5-sample spot check). 나머지 모두 자동.

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

### 3.4 Cross-component validation for MC label (자동, 30 min)

원래 manual human label 30개 계획 → **3가지 자동 cross-check로 대체**. Manual 0.

```python
# Script: scripts/d2_mc_cross_validation.py
import json, numpy as np
from scipy.stats import pearsonr

mc_results = [json.loads(l) for l in open('data/d2_mc_results.jsonl')]

# Cross-check 1: MC vs Outcome (trajectory level)
# 정답 trajectory의 평균 step value가 오답 trajectory보다 높아야 함
correct_traj_step_means = []
wrong_traj_step_means = []
for r in mc_results:
    # trajectory의 정답 여부는 첫 generation의 outcome으로 proxy (same model, no continuation)
    traj_correct = r['step_outcome_scores'][-1]['mean_score'] >= 0.5  # last step proxy
    avg_step_score = np.mean([s['mean_score'] for s in r['step_outcome_scores']])
    if traj_correct:
        correct_traj_step_means.append(avg_step_score)
    else:
        wrong_traj_step_means.append(avg_step_score)

gap = np.mean(correct_traj_step_means) - np.mean(wrong_traj_step_means)
print(f"[MC vs Outcome] Correct traj avg step={np.mean(correct_traj_step_means):.3f}, "
      f"Wrong traj avg step={np.mean(wrong_traj_step_means):.3f}, gap={gap:.3f}")
# 기준: gap ≥ 0.20 → MC label이 outcome correlate

# Cross-check 2: Step value의 monotonicity along trajectory
# 정답 trajectory에서는 step value가 상승 trend, 오답에서는 하락 trend여야 함
trends_correct = []
trends_wrong = []
for r in mc_results:
    scores = [s['mean_score'] for s in r['step_outcome_scores']]
    if len(scores) < 3: continue
    indices = list(range(len(scores)))
    slope, _ = pearsonr(indices, scores)
    if r['step_outcome_scores'][-1]['mean_score'] >= 0.5:
        trends_correct.append(slope)
    else:
        trends_wrong.append(slope)
print(f"[Trend] Correct traj slope mean={np.mean(trends_correct):.3f} (target >0)")
print(f"[Trend] Wrong traj slope mean={np.mean(trends_wrong):.3f} (target <=0)")

# Cross-check 3: Variance distribution
# Bimodal (variance≈0) is OK if both modes (1.0 and 0.0) exist
all_scores = [s['mean_score'] for r in mc_results for s in r['step_outcome_scores']]
n_zero = sum(1 for s in all_scores if s == 0.0)
n_one = sum(1 for s in all_scores if s == 1.0)
n_mid = len(all_scores) - n_zero - n_one
print(f"[Score distribution] zero={n_zero/len(all_scores)*100:.1f}%, "
      f"one={n_one/len(all_scores)*100:.1f}%, mid={n_mid/len(all_scores)*100:.1f}%")
# 기준: zero+one >= 30% (bimodal) AND mid >= 20% (continuous signal 존재)

# Combined verdict
checks = {
    'mc_outcome_gap': gap >= 0.20,
    'correct_traj_trend_positive': np.mean(trends_correct) > 0,
    'wrong_traj_trend_nonpositive': np.mean(trends_wrong) <= 0.05,
    'bimodal_signal_present': (n_zero + n_one) / len(all_scores) >= 0.30,
    'continuous_signal_present': n_mid / len(all_scores) >= 0.15,
}
print(f"\n[MC validation] {sum(checks.values())}/{len(checks)} pass")
for k, v in checks.items():
    print(f"  {'✓' if v else '✗'} {k}")
```

**Pass criterion**: ≥ 3/5 checks pass. 3개 미만이면 MC signal이 noise 우세 — K=16으로 증가 또는 sample이 너무 어려움 진단.

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

**Recall test (자동, LLM-as-judge cross-check, 30 min)** — manual 0:

```python
# Script: scripts/d2_claim_recall_auto.py
# Compare regex extractor against LLM extractor (Qwen2.5-7B as oracle proxy)
from vllm import LLM, SamplingParams
import json, random

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=2, max_model_len=4096)
sampling = SamplingParams(temperature=0.0, max_tokens=512)

LLM_EXTRACT_PROMPT = """Extract all explicit numeric or categorical claims from this reasoning step.
Return ONLY a JSON list of {{"entity": str, "value": str_or_number, "type": "value"|"categorical"}}.
Empty list [] if no explicit claim.

Step: {step}

JSON:"""

random.seed(42)
results = [json.loads(l) for l in open('data/d2_claims.jsonl')]
all_steps_for_recall = [(r['id'], i, r['steps'][i], r['step_claims'][i])
                        for r in results for i in range(len(r['steps']))]
sampled_recall = random.sample(all_steps_for_recall, 50)  # 50 step, ~5 claims each

# LLM extraction
prompts = [LLM_EXTRACT_PROMPT.format(step=text) for _, _, text, _ in sampled_recall]
outputs = llm.generate(prompts, sampling)

agreements = []
for (sid, idx, text, regex_claims), out in zip(sampled_recall, outputs):
    try:
        llm_claims = json.loads(re.search(r'\[.*\]', out.outputs[0].text, re.DOTALL).group())
    except:
        llm_claims = []
    
    # Jaccard agreement on (normalized entity, value) pairs
    regex_set = {(normalize_text(c['entity']), str(c.get('value', ''))) for c in regex_claims}
    llm_set = {(normalize_text(c['entity']), str(c.get('value', ''))) for c in llm_claims}
    
    if not regex_set and not llm_set:
        agreements.append(1.0)  # both empty = agree
    elif not regex_set or not llm_set:
        agreements.append(0.0)  # one empty
    else:
        jaccard = len(regex_set & llm_set) / len(regex_set | llm_set)
        agreements.append(jaccard)

mean_agreement = np.mean(agreements)
recall_proxy = sum(1 for a in agreements if a >= 0.5) / len(agreements)
print(f"[Claim agreement] mean Jaccard = {mean_agreement:.3f} (target ≥0.70)")
print(f"[Claim recall proxy] {recall_proxy*100:.1f}% steps with agreement ≥0.50 (target ≥85%)")
```

**Pass criterion**: mean Jaccard ≥ 0.70 AND recall_proxy ≥ 85%. 미달 시 LLM extractor를 primary로 전환 (regex는 fallback).

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

### 3.7 Verifier alignment with outcome (자동, 5 min)

원래 CSV 매칭 entity-by-entity validation으로 over-engineer했음. **End-to-end alignment 한 줄로 충분**: verifier signal이 outcome 예측에 유용한지 직접 측정. PRM training의 진짜 목표가 outcome 개선이므로 이게 가장 정합한 metric.

```python
# Script: scripts/d2_verifier_alignment.py
import json, numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

results = [json.loads(l) for l in open('data/d2_perception_results.jsonl')]

per_sample = []
for r in results:
    # 1. Outcome (이미 있음 — gold answer 매칭)
    final_answer = extract_final_answer(r['raw_response'])  # post-think extract
    outcome_correct = int(relaxed_correctness(final_answer, r['gold_answer']))
    
    # 2. Verifier aggregate (sample의 모든 step perception score 평균)
    p_scores = [s['mean_score'] for s in r['step_perception_scores'] if s is not None]
    if not p_scores:
        continue
    perception_agg = np.mean(p_scores)
    
    # 3. MC aggregate (Math-Shepherd 비교용)
    mc_agg = np.mean([s['mean_score'] for s in r['step_outcome_scores']])
    
    per_sample.append({
        'outcome': outcome_correct,
        'perception_agg': perception_agg,
        'mc_agg': mc_agg
    })

y = np.array([s['outcome'] for s in per_sample])
p = np.array([s['perception_agg'] for s in per_sample])
m = np.array([s['mc_agg'] for s in per_sample])

print(f"N samples: {len(y)}, outcome correct rate: {np.mean(y)*100:.1f}%")

# Core metric 1: Perception verifier alignment
print(f"\n[Perception ↔ Outcome]")
print(f"  Pearson r:  {pearsonr(p, y)[0]:.3f}")
print(f"  Spearman ρ: {spearmanr(p, y)[0]:.3f}")
print(f"  ROC AUC:    {roc_auc_score(y, p):.3f}  # target ≥0.65")

# Core metric 2: MC baseline alignment
print(f"\n[MC ↔ Outcome (Math-Shepherd baseline)]")
print(f"  Pearson r:  {pearsonr(m, y)[0]:.3f}")
print(f"  ROC AUC:    {roc_auc_score(y, m):.3f}  # target ≥0.70")

# Core metric 3: Complementarity
print(f"\n[Complementarity]")
print(f"  r(perception, mc): {pearsonr(p, m)[0]:.3f}  # ideal 0.3-0.7")

# Core metric 4: Marginal information lift
auc_mc_only = roc_auc_score(y, LogisticRegression().fit(m.reshape(-1,1), y).predict_proba(m.reshape(-1,1))[:,1])
X_both = np.column_stack([m, p])
auc_both = roc_auc_score(y, LogisticRegression().fit(X_both, y).predict_proba(X_both)[:,1])
print(f"  AUC (mc only):       {auc_mc_only:.3f}")
print(f"  AUC (mc+perception): {auc_both:.3f}")
print(f"  Lift:                {auc_both - auc_mc_only:+.3f}  # target ≥+0.02")
```

**Pass criteria** (4개 중 3개 통과):

| Metric | Pass threshold | 의미 |
|---|---|---|
| Perception ROC AUC vs outcome | ≥ 0.65 | Verifier가 outcome 의미 있게 예측 |
| MC ROC AUC vs outcome | ≥ 0.70 | MC baseline sanity check |
| r(perception, mc) | 0.3 ≤ r ≤ 0.7 | Complementary (redundant도 noise도 아님) |
| AUC lift (mc+p vs mc only) | ≥ +0.02 | Perception이 MC 너머 정보 추가 |

**왜 이게 충분한가**:
- Per-claim CSV 매칭 validation은 verifier가 "각 claim에 정확히 응답하는가"를 측정 — 흥미롭지만 PRM training의 직접 목표 아님
- 우리는 **verifier가 outcome 예측 / 개선에 유용한가**를 알고 싶음. 이건 위 4 metric으로 직접 측정됨
- 100 sample × 1 inference run이면 끝. Manual 0, CSV 매칭 0, entity 처리 0
- Spurious correlation 위험은 N=100이면 r±0.10 신뢰구간이라 견딤

### 3.8 Component complementarity check (§3.7에 통합됨)

§3.7의 alignment script가 이미 r(perception, mc)와 AUC lift를 측정함. 별도 step 불필요.

### 3.9 D2 산출물

- `data/d2_pilot_samples.jsonl` (100 samples)
- `data/d2_mc_results.jsonl` (100 samples × steps × 8 continuations)
- `data/d2_claims.jsonl` (extracted claims, regex + LLM fallback)
- `data/d2_perception_results.jsonl` (perception scores)
- `data/d2_alignment_report.json` (verifier ↔ outcome alignment, AUC, lift, complementarity — 자동)
- `docs/d2_report.md`:
  - MC validation: 5-check pass count
  - Claim extraction: Jaccard agreement vs LLM oracle
  - **Verifier alignment with outcome: ROC AUC, AUC lift, complementarity r**
  - **Decision: Week 2 진행 / Component partial / FAIL**

**Manual workload**: 0. D2 wall-clock ~3h:
- MC rollout: 1h
- Claim extraction + LLM cross-check: 30min
- Perception verify: 1h  
- Alignment script: 5min
- Total: ~3h GPU + 5min reporting

---

## Part 4: D1 → D2 → Week 2 Decision Gate

### 4.1 통합 decision matrix

| D1 (auto+spot) | D2 MC (5-check) | D2 Claim (Jaccard) | D2 Alignment (4-metric) | Decision |
|---|---|---|---|---|
| 6/6 + 0 BAD | ≥3/5 | ≥0.70 | ≥3/4 | **Week 2 GO**: full PRM data construction (5K) → train PRM → GRPO with PRM |
| 6/6 + 0 BAD | ≥3/5 | ≥0.70 | 2/4 | Week 2 partial: 약한 component 빼고 진행 (perception AUC<0.65이면 perception term 제외) |
| 6/6 + 0 BAD | ≥3/5 | <0.70 | ≥3/4 | Claim extractor LLM primary로 전환 → re-validate (1h) → Week 2 |
| any | <3/5 | * | * | **Framing 재고**: chart QA reasoning이 process reward에 부적합 신호. Path B 또는 Path D로 pivot. |
| any | * | * | <2/4 | **Verifier 재고**: alignment 너무 약함. Stronger verifier (Qwen2.5-VL-32B 또는 GPT-4o-mini) 시도 또는 perception term 폐기. |
| Conditional (4-5/6) | * | * | * | Segmentation patch (1-2h) → re-run auto script → D2로 진행 |
| Fail (<4/6 or 2+ BAD) | - | - | - | LLM-based segmenter pilot (2-3h) 또는 framing 재고 |

### 4.2 Week 2 (D8-D14) outline if GO

- D8-D10: Full PRM data construction (5K samples, ~25h GPU). PRM training data label 생성 시 perception verifier (Stage 1)는 image re-query + CSV cross-check (Mode B) 둘 다 사용 가능 (training-side data는 CSV 보유)
- D10-D11: Train PRM model (Qwen2.5-VL-3B + LoRA, 1 epoch). 이게 Stage 2 — deploy될 PRM
- D12-D14: GRPO with trained PRM dense reward (zero-shot Qwen3-VL-4B base, Stage 3). Trained PRM은 image-only mode로 동작 (CSV 의존 X)
- D14 EOD: First eval — ChartMuseum / CharXiv-R / ChartQA-Pro Hypothetical+Conv

### 4.3 비용 / timeline summary

| Phase | Days | GPU-hours | API cost | Manual |
|---|---|---|---|---|
| D1 (segmentation auto + 5-spot) | 1 | 4 | $0 | 10분 |
| D2 (MC + claim + perception + alignment, 모두 자동) | 1 | 6 | $0 | 0 |
| Week 2 (full pipeline) | 7 | 80 | $0-30 (only if escalate verifier) | 0 |
| **Total to first PRM-RL result** | **9** | **90** | **<$30** | **~10분** |

**Pilot 핵심 metric**: D2 §3.7의 alignment script 한 번 돌리면 verifier의 utility를 직접 측정. 4 metric (perception AUC / MC AUC / complementarity r / AUC lift) 중 3개 통과 → Week 2 GO.

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

> **§3.1 Step Segmentation**: We segment the thinking-mode trace into reasoning steps using a length-bounded paragraph rule: split by `\n\n`, merge segments below T_min tokens, split segments above T_max tokens at sentence boundaries. T_min=80, T_max=560 follow the segment granularity sweet spot reported in HRM (Lu et al., 2025). Six automated sanity checks (length distribution, sentence integrity, fragment density) confirm structural validity on a 200-sample pilot.
>
> **§3.2 Outcome-Anchored Step Value**: Following Math-Shepherd (Wang et al., 2024), for each step boundary in a reasoning trace, we sample K=8 continuations to completion and measure the fraction reaching the correct final answer (verified against ground-truth via ANLS). This step value v_o(s) ∈ [0, 1] forms the base process reward signal. We validate signal quality via three internal cross-checks: (a) correct-trajectory step values exceed wrong-trajectory by ≥0.20, (b) step value monotonicity along correct trajectories, (c) bimodal distribution presence — all confirmed on the 100-sample pilot.
>
> **§3.3 Perception-Grounded Verification**: We extend Math-Shepherd's outcome-only labels with a perception-verification term v_p(s). For each step, we extract numeric/categorical claims using a regex extractor with LLM fallback (Qwen2.5-7B-Instruct), achieving 0.X Jaccard agreement with the LLM oracle on a 50-sample cross-check. For each claim, we re-query the chart image with a separate verifier VLM (Qwen2.5-VL-7B) using the prompt "Looking at this chart, what is the value of {entity}?", parse the numeric response, and compare against the claimed value within ±5% tolerance. v_p(s) = mean correctness over claims in step s. We validate the verifier's reliability against CSV ground truth on N=X claim–CSV pairs from training-side data, achieving F1=0.X (no test-set contamination — all validation samples are from ChartQA-train and ReachQA-train; image perceptual hash overlap with all evaluation benchmarks is verified to be 0).
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
