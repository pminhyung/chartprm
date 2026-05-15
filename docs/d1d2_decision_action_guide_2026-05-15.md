# D1+D2 Decision Action Guide — Open-Weights Only

_Date: 2026-05-15_
_Author: research-strategy-advisor_
_Successor to: `chartvcr_verifier_design_and_d1d2_action_guide.md` (D2 §3.7 자동화 patch 후)_
_Predecessor failure: `perception_verification_failure_analysis_2026-05-15.md` (4B+9B 및 27B+27B 양쪽에서 alignment 1/4 실패)_

---

## 0. 컨텍스트 (research agent 필독)

이 가이드는 **D2 pilot이 4B+9B/27B+27B 양쪽에서 perception verifier alignment 실패** (AUC 0.568, lift +0.003) 후 작성된 **수정된 실험 액션**이다. Failure analysis (`perception_verification_failure_analysis_2026-05-15.md`)에서 Spec A+B+C+D를 제안받았고, 본 가이드는 그를 **open-weights only (API $0)** 환경에서 수행하기 위한 구체 spec.

### 0.1 핵심 변경사항 vs 원래 D1+D2 가이드

| 항목 | 원래 spec | 수정 spec |
|---|---|---|
| Verifier model | Qwen2.5-VL-7B (same family) | **InternVL2.5-26B (cross-family)** |
| Claim extractor | regex 4 patterns + LLM fallback | **LLM-primary (Qwen3.5-VL-27B with strict entity validation)** |
| Verifier prompt | "what is value of {entity}?" | **NOT_FOUND fallback prompt** |
| Tolerance | 5% / 15% binary | **10% / 25% relaxed** |
| Validation metric | perception AUC vs outcome | **4-cell drift × outcome distribution** |
| Baseline pool | (잘못 포함됨) v8 SFT | **zero-shot 4B / Chart-R1 7B / ChartGemma 12B (3 model)** |

### 0.2 절대 사용 금지

- v8 SFT, v9 SFT, v_hq_lite_sft (이전 세션에서 net-negative로 폐기 결정)
- ChartQA-Pro, ChartMuseum, CharXiv-Reasoning **test bench** (training-side data만 사용)
- 유료 API (GPT-4o, GPT-4o-mini, Claude API 등)

### 0.3 Goal — 2일 후 무엇이 결정되는가

**Day 2 EOD에 4-cell distribution 표 1장**으로 다음 셋 중 하나 결정:

- **Pattern A (Phase 2 GO)**: 모든 baseline에서 "shortcut" cell ≥15% → drift framing universal, PRM-RL 학습 launch
- **Pattern B (MC-only pivot)**: baseline 분포 차이 거의 없음 → Math-Shepherd MC를 single contribution으로 단순화
- **Pattern C (재고)**: ChartGemma의 grounded+correct이 우리 zero-shot보다 유의하게 높음 → 우리 차별화 약화, framing 재검토

---

## 1. Day 1 — Verifier Patch + Reliability Check

### 1.1 Setup (1h, GPU 0)

#### 1.1.1 Model 다운로드 (백그라운드 병렬)

```bash
# Terminal 1
huggingface-cli download OpenGVLab/InternVL2_5-26B \
    --local-dir models/internvl2_5_26b

# Terminal 2 (병렬)
huggingface-cli download omlab/Chart-R1-Qwen2-VL-7B-Instruct \
    --local-dir models/chart_r1_7b 2>/dev/null \
    || huggingface-cli download {정확한 Chart-R1 HF id 확인 후 입력} \
       --local-dir models/chart_r1_7b

# Terminal 3 (병렬)
huggingface-cli download ahmed-masry/chartgemma \
    --local-dir models/chartgemma_12b
```

**주의**: Chart-R1 정확한 HF id가 변경되었을 수 있음. arxiv 2507.15509 paper의 official repo 확인 후 정확한 model id 입력. ChartGemma는 `ahmed-masry/chartgemma` 표준.

다운로드 진행 중 §1.2 작업.

#### 1.1.2 vLLM 서버 launch (Verifier만, Day 1용)

```bash
# InternVL2.5-26B verifier serving (TP=2 on 2 GPUs)
CUDA_VISIBLE_DEVICES=2,3 nohup python -m vllm.entrypoints.openai.api_server \
    --model models/internvl2_5_26b \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port 8100 \
    --trust-remote-code \
    --served-model-name internvl_verifier \
    > /tmp/vllm_verifier.log 2>&1 &
```

Ready 확인:
```bash
sleep 60 && curl -s http://localhost:8100/v1/models | jq
```

### 1.2 Spec A — LLM-primary claim extractor (1h)

**File**: `scripts/d2_claim_extract_v2.py`

```python
"""D2 claim extractor v2 — LLM-primary with strict entity validation.

Replaces regex Tier 1 from v1 with Qwen3.5-VL-27B prompt-based extraction.
Goal: spurious claim ratio 30-50% → <10%.
"""
import json, re, argparse
from openai import OpenAI

EXTRACT_PROMPT = """You are extracting structured claims from a chart-reasoning step.

Given the chart image and the reasoning step text below, extract ALL numeric or categorical claims that this step makes about VISIBLE CHART ENTITIES.

STRICT RULES:
1. Entity MUST be a label/category that visibly appears on the chart axes, legend, or data labels.
2. REJECT pronouns ("it", "the answer", "this value")
3. REJECT math operation names ("Difference", "Average", "Ratio", "Sum", "Total" — these are computed results, not chart entities)
4. REJECT positional descriptors ("rightmost bar", "bottom", "topmost", "second column")
5. REJECT comparative adjectives ("smaller value", "bigger", "smallest", "highest")
6. REJECT sentence fragments. Decompose "Mali is 146.58" → entity="Mali", value=146.58. NOT entity="Second lowest is Mali".
7. If you cannot verify the entity exists ON THE CHART (axis label, legend item, data label), OMIT that claim entirely.

Reasoning step:
\"\"\"
{step_text}
\"\"\"

Output ONLY a JSON list (no preamble, no explanation). Format:
[{{"entity": "exact chart label", "value": number_or_null, "type": "value" or "categorical"}}]

If no valid claim, output [].
JSON:"""

def extract_claims_llm(client, model, step_text, image_path):
    """Call vLLM-served LLM with chart image + step text."""
    import base64
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": EXTRACT_PROMPT.format(step_text=step_text)}
            ]
        }],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    
    # Robust JSON extraction
    try:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            claims = json.loads(m.group())
        else:
            claims = []
    except (json.JSONDecodeError, AttributeError):
        claims = []
    
    # Validate schema
    valid_claims = []
    for c in claims:
        if not isinstance(c, dict): continue
        if 'entity' not in c or not c['entity']: continue
        valid_claims.append({
            'entity': str(c['entity']).strip(),
            'value': c.get('value'),
            'type': c.get('type', 'value')
        })
    return valid_claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/d2_pilot/segmented_v3.jsonl')
    parser.add_argument('--output', default='data/d2_pilot/claims_v2.jsonl')
    parser.add_argument('--extractor_port', type=int, default=8200,
                        help='Qwen3.5-VL-27B vLLM port (이미 launched)')
    parser.add_argument('--extractor_model', default='qwen3_5_vl_27b')
    args = parser.parse_args()
    
    client = OpenAI(base_url=f"http://localhost:{args.extractor_port}/v1", api_key="dummy")
    
    samples = [json.loads(l) for l in open(args.input)]
    out = []
    for s in samples:
        s_out = {**s, 'step_claims': []}
        for step_text in s['steps']:
            claims = extract_claims_llm(client, args.extractor_model, step_text, s['image_path'])
            s_out['step_claims'].append(claims)
        out.append(s_out)
    
    with open(args.output, 'w') as f:
        for s in out:
            f.write(json.dumps(s) + '\n')
    
    # Sanity log
    n_claims = sum(len(c) for s in out for c in s['step_claims'])
    n_steps = sum(len(s['steps']) for s in out)
    print(f"Total claims: {n_claims} (avg {n_claims/n_steps:.2f}/step)")
    print(f"Empty step claim rate: {sum(1 for s in out for c in s['step_claims'] if not c)/n_steps*100:.1f}%")


if __name__ == '__main__':
    main()
```

**전제**: Qwen3.5-VL-27B이 이미 vLLM port 8200에 serving 중 (이전 D1+D2 setup에서). 만약 안 되어 있으면:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server \
    --model {Qwen3.5-VL-27B path} --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --port 8200 --trust-remote-code --served-model-name qwen3_5_vl_27b \
    > /tmp/vllm_extractor.log 2>&1 &
```

**실행**:
```bash
python scripts/d2_claim_extract_v2.py \
    --input data/d2_pilot/segmented_v3.jsonl \
    --output data/d2_pilot/claims_v2.jsonl \
    --extractor_port 8200
```

**예상 시간**: 100 sample × ~6 step × ~1s LLM call = ~10분 (vLLM batching 고려).

### 1.3 Spec B — Verifier with NOT_FOUND fallback (30분)

**File**: `scripts/d2_perception_verify_v2.py`

```python
"""D2 perception verifier v2 — NOT_FOUND fallback + cross-family verifier.

Replaces same-family Qwen2.5-VL-7B verifier with InternVL2.5-26B (cross-family).
Adds NOT_FOUND escape so spurious entities don't get random number penalties.
"""
import json, re, argparse, base64
from openai import OpenAI

VERIFY_VALUE_PROMPT = """Looking at this chart, locate the entity labeled exactly '{entity}' on the chart's axes, legend, or data labels.

If you find it, reply with ONLY its associated numeric value (no units, no explanation).

If '{entity}' does NOT appear as a chart label (it's a derived value, math result, pronoun, or not visible on the chart), reply EXACTLY with: NOT_FOUND

Do not infer or compute. Read directly from the chart.

Reply:"""

VERIFY_CATEGORICAL_PROMPT = """Looking at this chart, identify which entity matches the description '{entity}'.

If you find it among the chart labels, reply with the entity name only.

If no chart entity matches '{entity}' (it's vague, derived, or not visible), reply EXACTLY: NOT_FOUND

Reply:"""


def verify_claim(client, model, claim, image_path):
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    if claim.get('type') == 'value' and claim.get('value') is not None:
        prompt = VERIFY_VALUE_PROMPT.format(entity=claim['entity'])
    else:
        prompt = VERIFY_CATEGORICAL_PROMPT.format(entity=claim['entity'])
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        temperature=0.0,
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip()
    
    # NOT_FOUND fallback
    if 'NOT_FOUND' in raw.upper() or 'NOT FOUND' in raw.upper():
        return {'score': None, 'verifier_response': raw, 'matched': False}
    
    # Score computation
    if claim.get('type') == 'value':
        try:
            extracted = float(re.search(r'-?\d+\.?\d*', raw).group())
        except (AttributeError, ValueError):
            return {'score': None, 'verifier_response': raw, 'matched': False}
        
        rel_diff = abs(extracted - claim['value']) / max(abs(claim['value']), 1e-10)
        abs_diff = abs(extracted - claim['value'])
        # Spec C — relaxed tolerance
        if rel_diff <= 0.10 or abs_diff <= 1.0:
            return {'score': 1.0, 'verifier_response': raw, 'extracted': extracted}
        elif rel_diff <= 0.25:
            return {'score': 0.5, 'verifier_response': raw, 'extracted': extracted}
        else:
            return {'score': 0.0, 'verifier_response': raw, 'extracted': extracted}
    else:  # categorical
        norm_resp = re.sub(r'[^a-z0-9]', '', raw.lower())
        norm_claim = re.sub(r'[^a-z0-9]', '', str(claim['entity']).lower())
        if norm_resp == norm_claim or norm_resp in norm_claim or norm_claim in norm_resp:
            return {'score': 1.0, 'verifier_response': raw}
        else:
            return {'score': 0.0, 'verifier_response': raw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/d2_pilot/claims_v2.jsonl')
    parser.add_argument('--output', default='data/d2_pilot/perception_v2.jsonl')
    parser.add_argument('--verifier_port', type=int, default=8100)
    parser.add_argument('--verifier_model', default='internvl_verifier')
    args = parser.parse_args()
    
    client = OpenAI(base_url=f"http://localhost:{args.verifier_port}/v1", api_key="dummy")
    
    samples = [json.loads(l) for l in open(args.input)]
    out = []
    for s in samples:
        s_out = {**s, 'step_perception_scores': []}
        for step_idx, step_claims in enumerate(s['step_claims']):
            if not step_claims:
                s_out['step_perception_scores'].append(None)
                continue
            verifications = [verify_claim(client, args.verifier_model, c, s['image_path']) 
                             for c in step_claims]
            valid_scores = [v['score'] for v in verifications if v['score'] is not None]
            if not valid_scores:
                s_out['step_perception_scores'].append(None)  # all NOT_FOUND
                continue
            s_out['step_perception_scores'].append({
                'verifications': verifications,
                'valid_scores': valid_scores,
                'mean_score': sum(valid_scores) / len(valid_scores),
                'not_found_rate': sum(1 for v in verifications if v['score'] is None) / len(verifications)
            })
        out.append(s_out)
    
    with open(args.output, 'w') as f:
        for s in out:
            f.write(json.dumps(s) + '\n')
    
    # Sanity log
    all_pscores = [s['mean_score'] for r in out for s in r['step_perception_scores'] if s is not None]
    nf_rates = [s['not_found_rate'] for r in out for s in r['step_perception_scores'] if s is not None]
    print(f"Steps with verified claims: {len(all_pscores)}")
    print(f"Mean perception score: {sum(all_pscores)/len(all_pscores):.3f}")
    print(f"Mean NOT_FOUND rate per step: {sum(nf_rates)/len(nf_rates)*100:.1f}%")


if __name__ == '__main__':
    main()
```

**실행**:
```bash
python scripts/d2_perception_verify_v2.py \
    --input data/d2_pilot/claims_v2.jsonl \
    --output data/d2_pilot/perception_v2.jsonl \
    --verifier_port 8100
```

**예상 시간**: ~600 valid claim × 0.5s = ~5분 (실제 0.5-1h with batching overhead).

### 1.4 Verifier reliability check — CSV-based (30분)

**File**: `scripts/d1_verifier_reliability_csv.py`

```python
"""Day 1 verifier reliability check using ChartQA-train CSV ground truth.

For 50 ChartQA-train samples (with CSV), query verifier with each entity in CSV
and check if verifier's response matches CSV value. NO manual annotation needed.

Pass criterion: agreement ≥85%, parse coverage ≥80%.
"""
import json, csv, re, base64, random
from openai import OpenAI

VERIFY_PROMPT = """Looking at this chart, locate the entity labeled exactly '{entity}' on the chart.

If you find it, reply with ONLY its associated numeric value (no units, no explanation).
If '{entity}' is not visible on the chart, reply EXACTLY: NOT_FOUND

Reply:"""


def load_csv_entities(csv_path, max_entities=10):
    """Extract (entity, value) pairs from CSV. Limit per chart to avoid bias."""
    rows = list(csv.reader(open(csv_path, encoding='utf-8')))
    if not rows or len(rows) < 2: return []
    headers = rows[0]
    pairs = []
    
    if len(headers) == 2:
        # Long format
        for r in rows[1:]:
            if len(r) >= 2:
                try:
                    val = float(r[1].replace(',', '').replace('%', ''))
                    pairs.append((r[0], val))
                except ValueError:
                    continue
    else:
        # Wide format — use (col_header) as primary entity if cell has clear name
        for col_idx in range(1, len(headers)):
            for r in rows[1:]:
                if len(r) <= col_idx: continue
                try:
                    val = float(r[col_idx].replace(',', '').replace('%', ''))
                    # Combine row label + col header
                    entity = f"{r[0]} {headers[col_idx]}".strip()
                    pairs.append((entity, val))
                except (ValueError, IndexError):
                    continue
    
    random.shuffle(pairs)
    return pairs[:max_entities]


def main():
    random.seed(42)
    
    client = OpenAI(base_url="http://localhost:8100/v1", api_key="dummy")
    model = 'internvl_verifier'
    
    # Load ChartQA-train samples (50 random)
    chartqa_samples = [json.loads(l) for l in open('data/chartqa_train_with_csv.jsonl')]
    sampled = random.sample(chartqa_samples, 50)
    
    results = []
    for s in sampled:
        entities = load_csv_entities(s['csv_path'], max_entities=4)
        if not entities: continue
        
        with open(s['image_path'], 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        for entity, gt_value in entities:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": VERIFY_PROMPT.format(entity=entity)}
                    ]
                }],
                temperature=0.0,
                max_tokens=64,
            )
            raw = response.choices[0].message.content.strip()
            
            # Parse
            if 'NOT_FOUND' in raw.upper():
                results.append({'entity': entity, 'gt': gt_value, 'response': raw,
                                'extracted': None, 'agree': None, 'status': 'not_found'})
                continue
            try:
                extracted = float(re.search(r'-?\d+\.?\d*', raw).group())
            except (AttributeError, ValueError):
                results.append({'entity': entity, 'gt': gt_value, 'response': raw,
                                'extracted': None, 'agree': None, 'status': 'parse_fail'})
                continue
            
            rel_diff = abs(extracted - gt_value) / max(abs(gt_value), 1e-10)
            agree = rel_diff <= 0.10
            results.append({'entity': entity, 'gt': gt_value, 'response': raw,
                            'extracted': extracted, 'agree': agree, 
                            'rel_diff': rel_diff, 'status': 'ok'})
    
    # Stats
    ok = [r for r in results if r['status'] == 'ok']
    not_found = [r for r in results if r['status'] == 'not_found']
    parse_fail = [r for r in results if r['status'] == 'parse_fail']
    
    coverage = len(ok) / len(results) * 100
    agreement = sum(1 for r in ok if r['agree']) / len(ok) * 100 if ok else 0
    
    print(f"Total queries: {len(results)}")
    print(f"  OK (parsed): {len(ok)} ({coverage:.1f}%)")
    print(f"  NOT_FOUND: {len(not_found)} ({len(not_found)/len(results)*100:.1f}%)")
    print(f"  Parse fail: {len(parse_fail)}")
    print(f"\nVerifier reliability (agreement on parsed): {agreement:.1f}%")
    print(f"\nPASS criteria: coverage ≥80% AND agreement ≥85%")
    pass_check = coverage >= 80 and agreement >= 85
    print(f"VERDICT: {'PASS' if pass_check else 'FAIL'}")
    
    # Save
    with open('data/d1_verifier_reliability.json', 'w') as f:
        json.dump({
            'n_total': len(results),
            'coverage_pct': coverage,
            'agreement_pct': agreement,
            'pass': pass_check,
            'results': results
        }, f, indent=2)


if __name__ == '__main__':
    main()
```

**전제**: `data/chartqa_train_with_csv.jsonl`이 있어야 함. 형식:
```json
{"id": "chartqa_train_X", "image_path": "data/chartqa/train/png/X.png", "csv_path": "data/chartqa/train/tables/X.csv"}
```

만약 없으면 다음 한 줄 script로 생성:
```bash
python -c "
import os, json, glob
csv_dir = 'data/chartqa/train/tables'
img_dir = 'data/chartqa/train/png'
out = []
for csv_path in glob.glob(f'{csv_dir}/*.csv'):
    sid = os.path.splitext(os.path.basename(csv_path))[0]
    img = f'{img_dir}/{sid}.png'
    if os.path.exists(img):
        out.append({'id': f'chartqa_train_{sid}', 'image_path': img, 'csv_path': csv_path})
with open('data/chartqa_train_with_csv.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
print(f'Wrote {len(out)} samples')
"
```

**실행**:
```bash
python scripts/d1_verifier_reliability_csv.py
cat data/d1_verifier_reliability.json | jq '.coverage_pct, .agreement_pct, .pass'
```

**Decision**:
- PASS (coverage ≥80% AND agreement ≥85%) → §1.5 alignment metric으로 진행
- FAIL → §1.6 verifier escalation

### 1.5 Alignment metric 재측정 (5분)

**File**: `scripts/d2_alignment_v2.py`

```python
"""D2 alignment metric — perception verifier vs outcome.
Same as v1 §3.7 (already in markdown), but uses perception_v2.jsonl input.
"""
import json, numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def extract_final_answer(raw_response):
    """Extract post-think answer."""
    import re
    m = re.search(r'</think>\s*(.+?)(?:$|<\|)', raw_response, re.DOTALL)
    return m.group(1).strip() if m else raw_response.strip()


def relaxed_correctness(pred, gold):
    """Reuse existing scorer."""
    # Replace with project's actual scorer (lmms-eval relaxed_correctness or VLMEvalKit ANLS)
    if pred is None or gold is None: return 0
    pred_norm = str(pred).strip().lower().replace(',', '').replace('%', '').replace('$', '')
    gold_norm = str(gold).strip().lower().replace(',', '').replace('%', '').replace('$', '')
    try:
        p, g = float(pred_norm), float(gold_norm)
        return int(abs(p - g) / max(abs(g), 1e-10) <= 0.05)
    except ValueError:
        return int(pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm)


def main():
    results = [json.loads(l) for l in open('data/d2_pilot/perception_v2.jsonl')]
    
    per_sample = []
    for r in results:
        final = extract_final_answer(r.get('raw_response', ''))
        outcome = relaxed_correctness(final, r['gold_answer'])
        
        p_scores = [s['mean_score'] for s in r['step_perception_scores'] if s is not None]
        if not p_scores: continue
        perception_agg = np.mean(p_scores)
        
        mc_agg = np.mean([s['mean_score'] for s in r['step_outcome_scores']])
        
        per_sample.append({'outcome': outcome, 'perception_agg': perception_agg, 'mc_agg': mc_agg})
    
    y = np.array([s['outcome'] for s in per_sample])
    p = np.array([s['perception_agg'] for s in per_sample])
    m = np.array([s['mc_agg'] for s in per_sample])
    
    print(f"N: {len(y)}, outcome correct: {np.mean(y)*100:.1f}%")
    
    # Metrics
    perception_auc = roc_auc_score(y, p)
    mc_auc = roc_auc_score(y, m)
    pm_corr = pearsonr(p, m)[0]
    
    auc_mc_only = roc_auc_score(y, LogisticRegression().fit(m.reshape(-1,1), y).predict_proba(m.reshape(-1,1))[:,1])
    auc_both = roc_auc_score(y, LogisticRegression().fit(np.column_stack([m,p]), y).predict_proba(np.column_stack([m,p]))[:,1])
    lift = auc_both - auc_mc_only
    
    print(f"\n[Alignment metrics]")
    print(f"  Perception AUC vs outcome: {perception_auc:.3f} (target ≥0.65)")
    print(f"  MC AUC vs outcome:         {mc_auc:.3f} (target ≥0.70)")
    print(f"  r(perception, mc):          {pm_corr:+.3f} (target 0.3-0.7)")
    print(f"  AUC lift (mc+p vs mc):     {lift:+.3f} (target ≥+0.02)")
    
    pass_count = sum([
        perception_auc >= 0.65,
        mc_auc >= 0.70,
        0.3 <= pm_corr <= 0.7,
        lift >= 0.02
    ])
    print(f"\n  Pass: {pass_count}/4 (target ≥3 for traditional alignment)")
    print(f"\nNOTE: Even if alignment fails, proceed to Day 2 4-cell measurement —")
    print(f"      decoupling between perception and outcome IS the framing thesis.")
    
    json.dump({
        'perception_auc': float(perception_auc),
        'mc_auc': float(mc_auc),
        'r_p_m': float(pm_corr),
        'lift': float(lift),
        'pass_count': int(pass_count),
        'n_samples': len(y)
    }, open('data/d2_pilot/alignment_v2.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
```

**실행**:
```bash
python scripts/d2_alignment_v2.py
cat data/d2_pilot/alignment_v2.json
```

**Day 1 EOD report**: 
- Verifier reliability (CSV agreement %)
- Alignment 4-metric (이전 1/4 → 새 ?/4)
- 결정: alignment 개선 됐든 안 됐든 **무조건 Day 2로 진행** (4-cell metric이 진짜 결정 metric)

### 1.6 Verifier escalation (CSV reliability FAIL 시만)

만약 InternVL2.5-26B reliability <85%면:
1. **Molmo-7B-D 시도** (visual grounding 특화):
   ```bash
   huggingface-cli download allenai/Molmo-7B-D-0924 --local-dir models/molmo_7b_d
   # vLLM serving with --trust-remote-code
   ```
2. 같은 §1.4 reliability test 재실행
3. Molmo도 미달이면 **InternVL3-38B** (있으면) 또는 **Qwen2.5-VL-72B** (4×A100 TP=4)
4. 셋 다 미달이면 framing의 perception term 자체 폐기, MC-only로 pivot

---

## 2. Day 2 — 3 Baseline 4-Cell Distribution

### 2.1 Setup (1.5h)

#### 2.1.1 Baseline model serving

기존 verifier (8100) + extractor (8200)는 유지. 추가로 baseline 3개 띄움:

```bash
# Chart-R1 7B serving (1 GPU)
CUDA_VISIBLE_DEVICES=8 nohup python -m vllm.entrypoints.openai.api_server \
    --model models/chart_r1_7b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --port 8300 \
    --trust-remote-code \
    --served-model-name chart_r1 \
    > /tmp/vllm_chart_r1.log 2>&1 &

# ChartGemma 12B serving (1 GPU)
CUDA_VISIBLE_DEVICES=9 nohup python -m vllm.entrypoints.openai.api_server \
    --model models/chartgemma_12b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --port 8301 \
    --trust-remote-code \
    --served-model-name chartgemma \
    > /tmp/vllm_chartgemma.log 2>&1 &

# zero-shot Qwen3.5-VL-4B (이미 D2 pilot에서 inference 한 결과 재사용)
# raw_response가 data/d2_pilot/segmented_v3.jsonl에 있으므로 추가 inference 불필요
```

Ready 확인:
```bash
sleep 90
for port in 8100 8200 8300 8301; do
    echo "Port $port:"
    curl -s http://localhost:$port/v1/models | jq '.data[].id'
done
```

GPU 사용 현황:
- 2-3: InternVL2.5-26B verifier (TP=2)
- 4-7: Qwen3.5-VL-27B extractor (TP=4)
- 8: Chart-R1 7B
- 9: ChartGemma 12B
- 10-11: 여유 (필요 시 ceiling baseline용)

### 2.2 3 Baseline inference (2h)

**File**: `scripts/d2_baseline_inference.py`

```python
"""D2 — Generate inference outputs from 3 baselines on the same 100 D2 pilot samples.
Outputs: baseline_inference_{model}.jsonl
"""
import json, base64, argparse
from openai import OpenAI

INFERENCE_PROMPT_TEMPLATE = """Looking at this chart, answer the following question step by step in a <think> block, then provide the final answer.

Question: {question}

<think>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, choices=['chart_r1', 'chartgemma'])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--input', default='data/d2_pilot/d2_pilot_samples.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_tokens', type=int, default=4096)
    args = parser.parse_args()
    
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="dummy")
    samples = [json.loads(l) for l in open(args.input)]
    
    out = []
    for s in samples:
        with open(s['image_path'], 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": INFERENCE_PROMPT_TEMPLATE.format(question=s['question'])}
                    ]
                }],
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            raw_response = response.choices[0].message.content
        except Exception as e:
            raw_response = f"<error>{e}</error>"
        
        out.append({**s, 'raw_response': raw_response, 'baseline_model': args.model_name})
    
    with open(args.output, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(f"Wrote {len(out)} samples to {args.output}")


if __name__ == '__main__':
    main()
```

**실행**:
```bash
# Chart-R1 (~30-40min)
python scripts/d2_baseline_inference.py \
    --model_name chart_r1 --port 8300 \
    --output data/d2_pilot/baseline_chart_r1.jsonl

# ChartGemma (~30-40min)
python scripts/d2_baseline_inference.py \
    --model_name chartgemma --port 8301 \
    --output data/d2_pilot/baseline_chartgemma.jsonl

# zero-shot Qwen3.5-VL-4B → 이미 pilot에 있음 (재사용)
cp data/d2_pilot/segmented_v3.jsonl data/d2_pilot/baseline_zeroshot_4b.jsonl
```

### 2.3 Baseline pipeline 실행 (각 baseline에 segment + extract + verify) (3h)

각 baseline 모델의 reasoning trace에 동일한 segmentation + claim extraction + perception verification 적용:

```bash
# Chart-R1
python scripts/d1_segment.py \
    --input data/d2_pilot/baseline_chart_r1.jsonl \
    --output data/d2_pilot/baseline_chart_r1_segmented.jsonl \
    --target_min 80 --target_max 560

python scripts/d2_claim_extract_v2.py \
    --input data/d2_pilot/baseline_chart_r1_segmented.jsonl \
    --output data/d2_pilot/baseline_chart_r1_claims.jsonl \
    --extractor_port 8200

python scripts/d2_perception_verify_v2.py \
    --input data/d2_pilot/baseline_chart_r1_claims.jsonl \
    --output data/d2_pilot/baseline_chart_r1_perception.jsonl \
    --verifier_port 8100

# ChartGemma — 동일 절차
# (스크립트 반복, model 이름만 변경)

# zero-shot 4B — 기존 pilot 재사용 (perception_v2.jsonl 있음)
cp data/d2_pilot/perception_v2.jsonl data/d2_pilot/baseline_zeroshot_4b_perception.jsonl
```

### 2.4 4-cell distribution 측정 (10분)

**File**: `scripts/d2_4cell_distribution.py`

```python
"""D2 — Compute 4-cell drift × outcome distribution for each baseline.
Output: per-model breakdown table.
"""
import json, numpy as np

DRIFT_THRESHOLD = 0.5  # perception_agg >= 0.5 = low drift (grounded)


def extract_final_answer(raw_response):
    import re
    m = re.search(r'</think>\s*(.+?)(?:$|<\|)', raw_response, re.DOTALL)
    return m.group(1).strip() if m else raw_response.strip()


def relaxed_correctness(pred, gold):
    if pred is None or gold is None: return 0
    pred_norm = str(pred).strip().lower().replace(',', '').replace('%', '').replace('$', '')
    gold_norm = str(gold).strip().lower().replace(',', '').replace('%', '').replace('$', '')
    try:
        p, g = float(pred_norm), float(gold_norm)
        return int(abs(p - g) / max(abs(g), 1e-10) <= 0.05)
    except ValueError:
        return int(pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm)


def assign_cell(perception_agg, outcome):
    if perception_agg is None:
        return 'unknown'
    low_drift = perception_agg >= DRIFT_THRESHOLD
    if outcome == 1 and low_drift: return 'grounded_correct'
    if outcome == 1 and not low_drift: return 'shortcut'
    if outcome == 0 and low_drift: return 'careful_flawed'
    return 'hallucinated'


def analyze_baseline(model_name, perception_file):
    samples = [json.loads(l) for l in open(perception_file)]
    cells = []
    for s in samples:
        final = extract_final_answer(s.get('raw_response', ''))
        outcome = relaxed_correctness(final, s['gold_answer'])
        
        p_scores = [x['mean_score'] for x in s['step_perception_scores'] if x is not None]
        perception_agg = np.mean(p_scores) if p_scores else None
        
        cells.append(assign_cell(perception_agg, outcome))
    
    n = len(cells)
    breakdown = {
        'grounded_correct': cells.count('grounded_correct') / n * 100,
        'shortcut': cells.count('shortcut') / n * 100,
        'careful_flawed': cells.count('careful_flawed') / n * 100,
        'hallucinated': cells.count('hallucinated') / n * 100,
        'unknown': cells.count('unknown') / n * 100,
    }
    breakdown['outcome_correct_rate'] = (cells.count('grounded_correct') + cells.count('shortcut')) / n * 100
    breakdown['n_samples'] = n
    return breakdown


def main():
    baselines = [
        ('zeroshot_4b', 'data/d2_pilot/baseline_zeroshot_4b_perception.jsonl'),
        ('chart_r1_7b', 'data/d2_pilot/baseline_chart_r1_perception.jsonl'),
        ('chartgemma_12b', 'data/d2_pilot/baseline_chartgemma_perception.jsonl'),
    ]
    
    results = {}
    for name, path in baselines:
        try:
            results[name] = analyze_baseline(name, path)
        except Exception as e:
            print(f"FAILED for {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Print table
    print(f"\n{'Model':<20}{'GroundedCorrect':>17}{'Shortcut':>10}{'CarefulFlawed':>15}{'Hallucinated':>14}{'Unknown':>9}{'Outcome%':>10}")
    print("="*95)
    for name, b in results.items():
        if 'error' in b:
            print(f"{name:<20}  ERROR: {b['error']}")
            continue
        print(f"{name:<20}{b['grounded_correct']:>16.1f}%{b['shortcut']:>9.1f}%"
              f"{b['careful_flawed']:>14.1f}%{b['hallucinated']:>13.1f}%"
              f"{b['unknown']:>8.1f}%{b['outcome_correct_rate']:>9.1f}%")
    
    # Save
    json.dump(results, open('data/d2_pilot/4cell_distribution.json', 'w'), indent=2)
    
    # Pattern detection
    print("\n=== Pattern detection ===")
    valid = {k: v for k, v in results.items() if 'error' not in v}
    if len(valid) < 2:
        print("FAIL: <2 valid baselines, cannot determine pattern")
        return
    
    shortcut_rates = [v['shortcut'] for v in valid.values()]
    grounded_rates = [v['grounded_correct'] for v in valid.values()]
    
    universal_shortcut = all(s >= 15 for s in shortcut_rates)
    grounded_diff = max(grounded_rates) - min(grounded_rates)
    
    if universal_shortcut and grounded_diff < 10:
        pattern = "A — drift universal, framing live → Phase 2 GO"
    elif universal_shortcut and grounded_diff >= 10:
        if 'chartgemma_12b' in valid and valid['chartgemma_12b']['grounded_correct'] == max(grounded_rates):
            pattern = "C — ChartGemma's grounded SFT cascades to step level → 차별화 약함, 재고"
        else:
            pattern = "A-strong — drift universal AND model-dependent → Phase 2 GO with stronger contribution"
    else:
        pattern = "B — drift not universal, MC-only Math-Shepherd로 pivot"
    
    print(f"Pattern: {pattern}")
    json.dump({**results, 'pattern': pattern}, 
              open('data/d2_pilot/4cell_distribution.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
```

**실행**:
```bash
python scripts/d2_4cell_distribution.py
cat data/d2_pilot/4cell_distribution.json | jq '.pattern'
```

### 2.5 Day 2 EOD report

**File**: `docs/d1d2_decision_report.md` 작성 (research agent가 생성)

템플릿:

```markdown
# D1+D2 Decision Report — 2026-MM-DD

## Day 1 결과

### Verifier setup
- Model: InternVL2.5-26B (cross-family)
- Reliability test (CSV-based, 200 query):
  - Coverage: __%
  - Agreement: __%
  - PASS / FAIL

### Patched extractor + verifier alignment (변화)
- Before patch: AUC=0.568, lift=+0.003 (1/4 pass)
- After patch (v2):
  - Perception AUC: __
  - MC AUC: __
  - r(p, m): __
  - Lift: __
  - Pass: _/4
- 변화 분석: ...

## Day 2 결과 — 4-cell distribution

| Model           | GroundedCorrect | Shortcut | CarefulFlawed | Hallucinated | Outcome% |
|-----------------|----------------:|---------:|--------------:|-------------:|---------:|
| zeroshot 4B     |              ?% |       ?% |            ?% |           ?% |       ?% |
| Chart-R1 7B     |              ?% |       ?% |            ?% |           ?% |       ?% |
| ChartGemma 12B  |              ?% |       ?% |            ?% |           ?% |       ?% |

### Pattern: A / B / C
- Reasoning: ...
- Implications for Phase 2: ...

## Decision

[Phase 2 GO / MC-only Pivot / Reconsider]

이유: ...

## 추가 발견 사항 / 우려

...
```

---

## 3. Decision Gate (Day 2 EOD)

### 3.1 Pattern A — Phase 2 GO

조건:
- 모든 baseline에서 `shortcut ≥ 15%`
- baseline 간 grounded_correct 차이 < 10pp (drift universal)

**액션**: Week 2 PRM training data 구축 launch (D8-D10).

### 3.2 Pattern A-strong — Phase 2 GO with stronger contribution

조건:
- 모든 baseline에서 shortcut ≥ 15%
- AND baseline 간 grounded_correct 차이 ≥ 10pp (model-dependent variation)

이게 강한 evidence: drift는 universal하지만 method/training에 따라 정도 다름. 우리 method가 grounded_correct를 더 높일 것.

**액션**: Week 2 GO + paper §1에 "drift varies by training paradigm" sub-thesis 추가.

### 3.3 Pattern B — MC-only pivot

조건:
- baseline 분포가 거의 동일 (shortcut <10pp 차이)
- 또는 grounded_correct 차이 미미

**액션**: 
1. Perception term 폐기 또는 minor weight (w_p=0.1)
2. MC-only Math-Shepherd PRM이 main contribution
3. Paper narrative: "First chart-domain application of Math-Shepherd auto-labeling, with cross-benchmark generalization"
4. Novelty 약화하지만 executable. D&B track 안전.

### 3.4 Pattern C — Reconsider

조건:
- ChartGemma의 grounded_correct이 zero-shot 대비 유의하게 (≥10pp) 높음
- → answer-level grounded SFT가 step-level grounding으로 cascade되는 것 시사
- 우리 framing의 차별화 약화

**액션**:
1. ChartGemma의 SFT data 분석 — 무엇이 grounded인가?
2. 우리 차별화 angle 재정의: ChartGemma는 SFT-only, 우리는 RL with PRM. RL의 specific 효과로 framing 좁히기
3. 또는 "Step-level grounded SFT data 자동 합성" angle (ChartGemma과 다른 axis)

---

## 4. Reporting checklist (research agent → 멘토 보고 시)

D2 EOD에 다음을 함께 보고:

- [ ] `data/d1_verifier_reliability.json` — verifier 신뢰도
- [ ] `data/d2_pilot/alignment_v2.json` — alignment 4-metric (참고용, 통과 못 해도 OK)
- [ ] `data/d2_pilot/4cell_distribution.json` — **핵심 결정 데이터**
- [ ] `docs/d1d2_decision_report.md` — 위 §2.5 템플릿대로 작성
- [ ] (Optional) 5-10 sample qualitative inspection: 각 cell별 실제 trajectory 1-2개씩 보여주기 (paper §3 case study용 소스)

---

## 5. 시간 / GPU 비용 summary

| 작업 | GPU 사용 | 시간 | API |
|---|---|---|---|
| Day 1 verifier setup (InternVL2.5-26B TP=2) | GPU 2-3 | 1h setup | $0 |
| Day 1 extractor patch (Spec A) | GPU 4-7 (TP=4) | 1h | $0 |
| Day 1 verifier reliability (CSV) | GPU 2-3 | 30분 | $0 |
| Day 1 D2 pipeline 재실행 (extractor + verifier + alignment) | 위 GPU 공유 | 1.5h | $0 |
| Day 2 baseline serving (Chart-R1 + ChartGemma) | GPU 8, 9 | setup 30분 | $0 |
| Day 2 baseline inference (3 model × 100 sample) | 위 GPU 공유 | 2h | $0 |
| Day 2 perception pipeline 적용 (3 baseline) | 위 GPU 공유 | 3h | $0 |
| Day 2 4-cell distribution + 보고서 | CPU | 30분 | $0 |
| **Total** | **GPU 2-3, 4-7, 8, 9 (총 8 GPU)** | **~10h GPU** | **$0** |

GPU 0-1, 10-11 여유 (필요 시 ceiling baseline용).

---

## 6. Failure mode patches (research agent용)

### 6.1 vLLM serving 실패

```bash
# GPU memory 안 잡힐 때 zombie process 확인
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
# 멘토에게 PID 보고 → kill 명령 받기 (자동 kill 절대 X)
```

### 6.2 Chart-R1 / ChartGemma 다운로드 실패

- Chart-R1 정확한 HF id를 모를 경우: arxiv 2507.15509 paper 본문/abstract의 GitHub repo 확인 → README의 model card link
- 후보: `omlab/Chart-R1`, `omlab/ChartR1`, `Chart-R1/Qwen2-VL-7B` 등
- 1차 시도 실패 시 즉시 멘토 보고. 임의 가정 금지.

### 6.3 InternVL2.5-26B reliability < 85%

§1.6 escalation 따름. Molmo-7B-D 우선 시도, 안 되면 멘토 보고.

### 6.4 4-cell distribution에서 모든 cell이 한쪽으로 쏠림

예: 모든 모델이 grounded_correct = 90%+ → drift signal 거의 없음.
- DRIFT_THRESHOLD를 0.5에서 0.7로 올려 다시 분류
- 그래도 한쪽 쏠림이면 verifier가 너무 관대함 → tolerance 더 strict (10% → 5%)
- 멘토 보고 필요

### 6.5 Outcome rate가 baseline 간 너무 다름

예: Chart-R1 outcome=80%, ChartGemma=30%. 비교 fair한가?
- 답: shortcut/grounded_correct 절대 비율 자체보다 **각 baseline의 outcome correct subset 내에서 grounded_correct vs shortcut 비율**을 normalized로 본다.
- 추가 metric: `drift_rate_within_correct = shortcut / (grounded_correct + shortcut)` 계산해서 비교.

---

## 7. 즉시 시작 체크리스트

연구 agent가 본 가이드 받으면 순서대로:

1. [ ] HF download 3개 (InternVL2.5-26B, Chart-R1, ChartGemma) 백그라운드 시작
2. [ ] Qwen3.5-VL-27B vLLM 8200 (이미 있는지 확인, 없으면 launch)
3. [ ] InternVL2.5-26B vLLM 8100 launch (다운로드 완료 후)
4. [ ] §1.4 verifier reliability test 실행 → PASS 확인
5. [ ] §1.2 + §1.3 + §1.5 patched pipeline + alignment metric
6. [ ] Day 1 EOD interim 보고 (Slack or Markdown)
7. [ ] §2.1 Chart-R1 + ChartGemma vLLM serving
8. [ ] §2.2 + §2.3 baseline inference + pipeline
9. [ ] §2.4 4-cell distribution + pattern detection
10. [ ] §2.5 decision report markdown 작성
11. [ ] 최종 보고: §4 checklist 항목 모두 첨부

소요: 2일 (Day 1 ~6h, Day 2 ~7h, 보고서 1h).

---

_End of action guide. 질문 있으면 멘토에게._
