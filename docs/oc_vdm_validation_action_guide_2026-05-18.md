# OC-VDM Validation Action Guide — Tier 1-B Cross-Bench Image-Dependency

_Date: 2026-05-18_
_Author: research-strategy-advisor_
_Successor to: `process_reward_experiments_2026-05-18.md` (chartmuseum 11-sample preflight)_
_Replaces: `d1d2_decision_action_guide_2026-05-15.md` (perception verifier 패러다임 폐기)_

---

## 0. 컨텍스트 (research agent 필독)

### 0.1 폐기된 패러다임 (HIGH confidence negative)

- **VLM perception verifier 4-axis**: chartmuseum dead 11-sample에서 lenient bias (mean 0.91) 확인. Sharpening 후도 NA dominance (80%). 다양한 ablation 시도 모두 failure mode 동일.
- **Claim parsing 기반 step-content 평가**: chartmuseum_759 같은 free-form thinking trace에서 0 verifiable claim 가진 step 존재. Rule-based / LLM-based extractor 모두 broken.
- **Per-claim deterministic verification (CSV/metadata lookup)**: chart reasoning이 수치 외 categorical/comparative/ordering/visual 등 다양 claim type 섞임. Free-form text에서 안정적 parsing 불가능.

위 셋은 paper §5 limitations로 명시 (honest negative result로 정직 보고).

### 0.2 살아남은 design — OC-VDM (Outcome-Conditional Visual Dependency Modulation)

핵심 원리: **step content 안 봄. Outcome-based signal만 사용**.

```python
# Per step (boundary는 \n\n + length cap으로 rough 식별):
mc_with(step k)    = mean(K' sub-rollout outcomes WITH image)      # Math-Shepherd 정통
mc_without(step k) = mean(K' sub-rollout outcomes WITHOUT image)   # NEW (free verifier)
visual_dep(step k) = mc_with(k) − mc_without(k)                    # step-level perception signal

# Advantage modulation (outcome-conditional)
base_advantage = standard GRPO group-relative advantage from outcome
if base_advantage > 0:
    step_advantage = base_advantage × max(0.2, 1 + λ × vd)
else:
    step_advantage = base_advantage  # penalty unchanged
```

**Step boundary 정확도 요구사항**:
- ❌ Semantic boundary alignment (claim 단위 정확히 자르기) — 불필요
- ✅ Token-level valid prefix (LLM이 이어서 generate할 수 있는 지점) — 필요
- ✅ Reasonable granularity (5-8 step/sample 평균) — 권장

→ `\n\n` split + merge<80tok + split>560tok 표준 사용. Math-Shepherd / ProcessBench / VisualPRM 동일.

### 0.3 현재 validate 안 된 가설

OC-VDM이 `chartmuseum` 11-sample에서만 작동 확인. 다른 bench에서 universal한지 미검증.

**Tier 1-B의 핵심 목적**: 12 combo (3 bench × 4 model) × ~50 samples로 OC-VDM의 universal 적용 가능성 검증.

---

## 1. 검증할 4 가설 (각각 PASS/FAIL 기준)

### H1: Visual dependency signal이 모든 bench에서 의미 있게 분포

**검증**: 12 combo의 step-level vd distribution.

**PASS 기준**:
- 각 bench에서 `std(vd) ≥ 0.15` (의미 있는 spread)
- 각 bench에서 `|vd| > 0.3` step 비율 ≥ 20% (informative step 충분)

**FAIL 시**: vd가 모든 step에서 ≈ 0이면 OC-VDM이 chartmuseum-specific. Narrative pivot 필요.

### H2: Pattern A/B/C/D가 모든 bench에서 의미 있는 비율로 존재

**검증**: 12 combo의 4 outcome-based pattern 분포.

```
Pattern A (hard-impossible):    mc_with=0 AND mc_without=0
Pattern B (text leakage):       mc_without ≥ 0.5 AND vd < 0
Pattern C (image-critical):     mc_with ≥ 0.5 AND vd > 0.3
Pattern D (hard-perception):    0 < mc_with < 0.3 AND mc_without ≈ 0
```

**PASS 기준**:
- 모든 bench에서 Pattern A 비율 < 50% (학습 가능 sample 충분)
- 최소 2 bench에서 Pattern C 비율 ≥ 15% (image-critical step 충분)
- 최소 1 bench에서 Pattern B 비율 ≥ 5% (leakage filter 정당화)

**FAIL 시**: Pattern 분포가 chartmuseum과 다르면 design 재구성.

### H3: OC-VDM advantage modulation이 4 pattern 모두에서 의도대로 작동

**검증**: 기존 image_dep_dead 11 sample + Tier 1-B 600 sample에 OC-VDM advantage 함수 시뮬레이션 (학습 없이 advantage 계산만).

**PASS 기준**: verbatim check —
- Pattern A: advantage = 0 (filter 후 drop)
- Pattern B sample step: positive base advantage가 0.3-0.7× modulation 받음 (down-weight)
- Pattern C sample step: positive base advantage가 1.3-1.7× modulation 받음 (up-weight)
- Pattern D sample step: 약한 양수 modulation (1.0-1.2×)

**FAIL 시**: Modulation 함수 hyperparameter (λ, floor) 조정.

### H4: Filter rule (hard-impossible + leakage)이 학습 가능 sample 비율 적절

**검증**: 12 combo에서 filter 적용 후 keep된 sample 비율.

**PASS 기준**:
- 각 bench에서 keep 비율 ≥ 50% (충분한 학습 데이터 잔존)
- Filter dropped sample이 특정 question type/subset에 편중되지 않음 (sample stratified 확인)

**FAIL 시**: Filter threshold 조정 또는 filter 폐기.

---

## 2. 실험 spec

### 2.1 Tier 1-B: 12-combo image_dep measurement

**Input source**: `data/d2_hardbench/perception/<model>/<bench>.jsonl`
- 4 model × 3 bench = 12 combo
- 각 combo의 wrong sample (outcome=0)만 sample, 50개

**Sample selection rule**:
```python
for combo in 12_combos:
    samples = load(f'data/d2_hardbench/perception/{model}/{bench}.jsonl')
    wrong = [s for s in samples if compute_outcome(s) == 0]
    selected = wrong[:50]  # 우선 처음 50개, 부족하면 가능한 만큼
```

**측정 procedure** (per sample × per step):
- K_prime = 6 (Math-Shepherd 표준에 가까움, 비용 manageable)
- Step segmentation: 기존 `\n\n` split + merge<80tok + split>560tok 사용
- Per step:
  - mc_with: prefix + image, K'=6 sub-rollout
  - mc_without: prefix + image-hidden (text-only), K'=6 sub-rollout
- Outcome verifier: bench-specific (relaxed_correctness for chartqa_pro, judge LLM for chartmuseum/charxiv) — 기존 `score_subrollout` 재사용
- **Empty-pred bug fix 유지** (`if not pred.strip(): return 0`)

**비용 estimate**:
```
50 sample × ~6 step × (6 with + 6 without) = 3,600 sub-rollouts per combo
12 combos × 3,600 = 43,200 sub-rollouts
Avg ~1,500 token/sub-rollout = 64.8M tokens

8×A100 vLLM @ ~10K tok/sec aggregate = ~108 min = ~2시간 wall-clock
```

**Output**: `data/d2_hardbench/reports/image_dep_<model>_<bench>.jsonl` (12 files)
- Schema 동일하게 `image_dep_dead.jsonl` 형식 유지

### 2.2 Boundary sensitivity ablation (작은 sanity check)

**목적**: OC-VDM 결과가 step boundary 정의에 민감하지 않음 확인.

**실험**: chartmuseum 50 sample (선택 1 combo만) × 3 boundary 방식:
- `\n\n` split + length cap (default, target_min=80, target_max=560)
- Sentence boundary (finer)
- Fixed token window (200 tokens uniform)

**측정**: 각 boundary 방식에서 vd 분포, Pattern A/B/C/D 비율 비교.

**PASS 기준**: Pattern 비율이 boundary 방식 간 ±10pp 이내. → boundary 선택이 결과 driver 아님 확인.

**FAIL 시**: Boundary 선택이 결과에 큰 영향. Paper §3에 boundary spec 명시 + ablation table 필수.

비용: ~10분 (50 sample × 3 spec).

### 2.3 OC-VDM advantage simulation (학습 전 verbatim verification)

**Input**: 위 §2.1의 600 sample (Tier 1-B 출력)

**Spec** — pure advantage computation, 학습 0:
```python
def oc_vdm_advantage(sample, K_outer=8, lambda_vd=1.0):
    """
    Simulate OC-VDM advantage computation.
    K_outer: simulated GRPO group size (use mc_value as proxy for trajectory outcome variance)
    """
    step_advantages = []
    for k, sr in enumerate(sample['step_records']):
        mc_w = sr['mc_value']
        mc_wo = sr['mc_without_value']
        vd = mc_w - mc_wo
        
        # Proxy base_advantage: trajectory-level outcome relative to group baseline
        # For simulation, use mc_w directly as advantage proxy
        base_advantage = mc_w - 0.5  # assume group baseline ~ 0.5
        
        if base_advantage > 0:
            modulation = max(0.2, 1.0 + lambda_vd * vd)
            step_adv = base_advantage * modulation
        else:
            step_adv = base_advantage
        
        step_advantages.append({
            'k': k, 'mc_w': mc_w, 'mc_wo': mc_wo, 'vd': vd,
            'base_adv': base_advantage, 'modulated_adv': step_adv
        })
    return step_advantages
```

**Verbatim check**: 600 sample 중 각 pattern type별로 10 sample씩 골라 advantage 값 출력 → manual review:
- Pattern A step: advantage ≈ 0 (mc_w=0이므로 base 음수)
- Pattern B step: advantage modulation ≤ 1.0 (vd<0 → down-weight 시뮬)
- Pattern C step: advantage modulation ≥ 1.3 (vd>0.3 → up-weight 시뮬)
- Pattern D step: advantage 약한 양수 modulation

스크립트 길이: ~80 lines.

### 2.4 Filter rule application + stratification check

**Input**: §2.1 600 sample

**Filter rules** (이미 §1 H4에 정의):
```python
def filter_sample(sample):
    sr = sample['step_records']
    mc_w_max = max(r['mc_value'] for r in sr)
    mc_wo_max = max(r['mc_without_value'] for r in sr)
    
    if mc_w_max == 0 and mc_wo_max == 0:
        return 'drop_hard_impossible'
    if mc_wo_max >= 0.5 and mc_w_max < mc_wo_max + 0.1:
        return 'drop_leakage'
    if mc_w_max == 1 and all(r['mc_value'] >= 0.8 for r in sr):
        return 'drop_trivial'
    return 'keep'
```

**Measurement**: 12 combo 각각의 (kept, drop_hard, drop_leakage, drop_trivial) 비율 + question type/subset stratification.

**Output**: `data/d2_hardbench/reports/oc_vdm_filter_stats.json`

### 2.5 (조건부) Mini-GRPO training comparison

**조건**: §1 H1-H4 모두 PASS 시에만 진행.

**Setup**:
- Policy: Qwen3.5-VL-4B-Instruct
- Training data: §2.4 filter 통과한 600 sample 중 stratified 300 sample
- 4 design × 200 step training × 6 GPU-hour each = ~24 GPU-hour 총
- Eval: 3 hard bench full test set (ChartQA-Pro / CharXiv-R / ChartMuseum)

**4 design comparison**:
- R1: outcome-only GRPO (baseline)
- R2: + MS-MC step value (Math-Shepherd 정통)
- R3: + OC-VDM advantage modulation (R2 + visual_dep)
- R4: + OC-VDM + filter (R3 + dataset filter applied)

**PASS 기준**:
- R4 > R1 ≥ +2.5pp on AVG of 3 hard benches
- R4 > R2 ≥ +1.0pp (OC-VDM modulation 가치 입증)
- R4 > R3 (filter 가치 입증)

**FAIL 시**:
- R3 ≈ R2: OC-VDM modulation 효과 없음 → 폐기, Math-Shepherd-only로 narrow
- R4 < R3: Filter가 학습 데이터 부족 야기 → filter 완화

---

## 3. 구현 spec

### 3.1 Scripts (작성/수정 대상)

| 파일 | 역할 | 상태 |
|---|---|---|
| `scripts/image_dep_mc_v2.py` | Tier 1-B 12-combo measurement (기존 `image_dep_mc.py` 확장) | 신규 작성 |
| `scripts/boundary_sensitivity.py` | §2.2 ablation | 신규 작성 |
| `scripts/oc_vdm_advantage_sim.py` | §2.3 verbatim simulation | 신규 작성 |
| `scripts/oc_vdm_filter_apply.py` | §2.4 filter + stratification | 신규 작성 |
| `scripts/oc_vdm_analyze.py` | 12 combo cross-bench 통합 분석 | 신규 작성 |
| `train_grpo_dapo.py` | §2.5 OC-VDM advantage 통합 (조건부) | 수정 |

### 3.2 `image_dep_mc_v2.py` 핵심 변경점

```python
# 기존 image_dep_mc.py는 math_shepherd_dead.jsonl 입력 받음
# v2: perception/<model>/<bench>.jsonl 직접 입력, filter outcome=0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='perception/<model>/<bench>.jsonl 경로')
    parser.add_argument('--output', required=True)
    parser.add_argument('--K_prime', type=int, default=6)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--filter_outcome', type=int, default=0,
                        help='select samples with this outcome (0=wrong, 1=correct, -1=all)')
    parser.add_argument('--policy_port', type=int, default=8501)
    parser.add_argument('--judge_port', type=int, default=8000)
    parser.add_argument('--judge_host', default='10.1.211.147')
    args = parser.parse_args()
    
    samples = load_perception_jsonl(args.input)
    if args.filter_outcome != -1:
        # Compute outcome per sample
        for s in samples:
            s['_outcome'] = compute_outcome_from_response(s['raw_response'], s['gold_answer'], s['scoring'])
        samples = [s for s in samples if s['_outcome'] == args.filter_outcome]
    
    samples = samples[:args.limit]
    
    # 기존 image_dep_mc.py logic: 각 sample의 each step에 mc_with + mc_without 측정
    # ⚠️ Empty-pred bug fix 유지: if not pred.strip(): return 0
    
    results = []
    for s in samples:
        result = measure_image_dep(s, args.K_prime, policy_client, judge_client)
        results.append(result)
    
    save_jsonl(args.output, results)
```

**Launch all 12 combos**:
```bash
#!/bin/bash
mkdir -p data/d2_hardbench/reports/image_dep_12combo
for model in qwen3vl_4b qwen3vl_8b_thinking chart_r1 chartgemma; do
    for bench in chartqa_pro charxiv_reasoning chartmuseum; do
        python scripts/image_dep_mc_v2.py \
            --input data/d2_hardbench/perception/$model/$bench.jsonl \
            --output data/d2_hardbench/reports/image_dep_12combo/${model}_${bench}.jsonl \
            --K_prime 6 \
            --limit 50 \
            --filter_outcome 0 \
            --policy_port 8501 \
            --judge_host 10.1.211.147 \
            --judge_port 8000
    done
done
```

### 3.3 `oc_vdm_analyze.py` 핵심 출력

```python
def analyze_12_combo(reports_dir):
    """
    Output:
    - data/d2_hardbench/reports/oc_vdm_cross_bench.json
    """
    results = {}
    for model in MODELS:
        for bench in BENCHES:
            path = f'{reports_dir}/{model}_{bench}.jsonl'
            samples = load_jsonl(path)
            
            # Collect all step-level vd
            all_vd = [r['visual_dep'] for s in samples for r in s['step_records']]
            
            # Pattern classification per step
            patterns = []
            for s in samples:
                for r in s['step_records']:
                    mc_w, mc_wo, vd = r['mc_value'], r['mc_without_value'], r['visual_dep']
                    if mc_w == 0 and mc_wo == 0:
                        p = 'A_hard_impossible'
                    elif mc_wo >= 0.5 and vd < 0:
                        p = 'B_leakage'
                    elif mc_w >= 0.5 and vd > 0.3:
                        p = 'C_image_critical'
                    elif 0 < mc_w < 0.3 and mc_wo < 0.1:
                        p = 'D_hard_perception'
                    else:
                        p = 'E_other'
                    patterns.append(p)
            
            # Sample-level filter outcomes
            filter_outcomes = [filter_sample(s) for s in samples]
            
            results[f'{model}__{bench}'] = {
                'n_samples': len(samples),
                'n_steps': len(all_vd),
                'vd_std': np.std(all_vd),
                'vd_informative_pct': sum(1 for v in all_vd if abs(v) > 0.3) / len(all_vd) * 100,
                'pattern_distribution': Counter(patterns),
                'filter_outcomes': Counter(filter_outcomes),
                'kept_pct': filter_outcomes.count('keep') / len(filter_outcomes) * 100,
            }
    
    json.dump(results, open(f'{reports_dir}/oc_vdm_cross_bench.json', 'w'), indent=2)
    
    # Pretty print decision-relevant metrics
    print_decision_table(results)
```

**Pretty print example**:
```
Combo                                      n_steps  vd_std  vd_info%  PatA%  PatB%  PatC%  PatD%  Kept%
qwen3vl_4b__chartqa_pro                       312    0.18    24.0%   12.5%   3.2%  18.3%   9.4%  68.0%
qwen3vl_4b__charxiv_reasoning                 285    0.21    28.4%    8.8%   4.6%  21.4%  11.6%  72.0%
qwen3vl_4b__chartmuseum                       312    0.22    31.1%   15.4%   8.7%  16.0%  14.1%  56.0%
qwen3vl_8b_thinking__chartqa_pro              ...
...
```

→ Decision gate에 직접 사용.

### 3.4 `oc_vdm_advantage_sim.py` 핵심 검증

```python
def simulate_advantage_per_pattern():
    """
    For each pattern, sample 10 steps and print advantage modulation.
    Verifies design works as intended.
    """
    samples = load_all_12_combo()
    by_pattern = defaultdict(list)
    
    for s in samples:
        for r in s['step_records']:
            pattern = classify_pattern(r)
            by_pattern[pattern].append((s['id'], r))
    
    for pattern, steps in by_pattern.items():
        print(f"\n=== Pattern {pattern} (n={len(steps)}) ===")
        for sid, sr in steps[:10]:
            base_adv = sr['mc_value'] - 0.5  # group baseline proxy
            vd = sr['visual_dep']
            
            if base_adv > 0:
                modulation = max(0.2, 1.0 + 1.0 * vd)
                modulated = base_adv * modulation
            else:
                modulation = 1.0  # no modulation for penalty
                modulated = base_adv
            
            print(f"  {sid}: mc_w={sr['mc_value']:.2f}, mc_wo={sr['mc_without_value']:.2f}, "
                  f"vd={vd:+.2f}, base={base_adv:+.2f}, mod={modulation:.2f}×, "
                  f"final={modulated:+.2f}")
    
    # Aggregate: mean modulation per pattern
    print("\n=== Mean modulation per pattern ===")
    for pattern, steps in by_pattern.items():
        mods = []
        for sid, sr in steps:
            base_adv = sr['mc_value'] - 0.5
            if base_adv > 0:
                mods.append(max(0.2, 1.0 + 1.0 * sr['visual_dep']))
            else:
                mods.append(1.0)
        print(f"  {pattern}: mean mod = {np.mean(mods):.3f}, n={len(mods)}")
```

**Decision verbatim check 후 다음으로 진행**:
- Pattern C 평균 mod ≥ 1.3 → OK
- Pattern B 평균 mod ≤ 0.8 → OK
- Pattern A advantage ≈ 0 → OK (mc_value=0이라 base_adv 음수, modulation 안 됨)
- 미달이면 λ 조정 (default 1.0 → 0.5 또는 1.5)

### 3.5 `train_grpo_dapo.py` 통합 (§2.5 진행 시)

기존 `reward_vapv_v2` 폐기. 새 advantage computation:

```python
def compute_oc_vdm_advantages(rollouts, lambda_vd=1.0, filter_apply=True):
    """
    rollouts: list of trajectories, each with:
        - step_records: [{mc_value, mc_without_value, ...}]
        - final_outcome: 0 or 1
        - token_step_map: {token_idx: step_k}
    """
    # Pre-filter samples
    if filter_apply:
        rollouts = [r for r in rollouts if filter_sample(r) == 'keep']
    
    # Group-relative outcome advantage (GRPO 표준)
    outcomes = [r['final_outcome'] for r in rollouts]
    group_mean = np.mean(outcomes)
    group_std = np.std(outcomes)
    
    token_advantages = {}
    for traj_idx, traj in enumerate(rollouts):
        base_traj_advantage = (outcomes[traj_idx] - group_mean) / max(group_std, 1e-8)
        
        for k, sr in enumerate(traj['step_records']):
            vd = sr['mc_value'] - sr['mc_without_value']
            
            if base_traj_advantage > 0:
                modulation = max(0.2, 1.0 + lambda_vd * vd)
                step_adv = base_traj_advantage * modulation
            else:
                step_adv = base_traj_advantage
            
            # Assign step advantage to all tokens in this step
            for token_idx in traj['token_step_map'].get(k, []):
                token_advantages[(traj_idx, token_idx)] = step_adv
    
    return token_advantages
```

코드 양: ~70 lines 추가. 기존 reward 함수 교체.

---

## 4. Decision gates

### Gate 1 (Day 1 EOD) — Tier 1-B measurement 완료 후

읽을 file: `data/d2_hardbench/reports/oc_vdm_cross_bench.json`

| 조건 | 다음 단계 |
|---|---|
| H1, H2 모두 PASS | §2.3 advantage simulation 진행 |
| H1 FAIL (vd 분포 좁음) on 모든 bench | OC-VDM은 chartmuseum-only → Math-Shepherd 단독으로 narrow, design 재구성 |
| H1 PASS but H2 FAIL (Pattern C/B 부족) | OC-VDM modulation 효과 적음, filter만 사용하는 narrow design 검토 |
| Sample 측정 실패 (judge crash 등) | bug fix 후 재실행, 멘토 report |

### Gate 2 (Day 2 EOD) — Advantage simulation 후

읽을 file: `oc_vdm_advantage_sim.py` 출력

| 조건 | 다음 단계 |
|---|---|
| H3 PASS (4 pattern 의도대로 작동) | §2.4 filter stratification 진행 |
| H3 PARTIAL (1-2 pattern 의도와 다름) | λ 조정 후 재시뮬 (1.0 → 0.5 또는 1.5) |
| H3 FAIL (modulation 효과 미미) | OC-VDM 폐기, Math-Shepherd 단독 narrow design |

### Gate 3 (Day 3 EOD) — Filter stratification 후

읽을 file: `oc_vdm_filter_stats.json`

| 조건 | 다음 단계 |
|---|---|
| H4 PASS (kept ≥ 50%, stratified balance OK) | §2.5 mini-GRPO training 진행 |
| Kept < 50% on 1+ bench | Filter threshold 완화 (mc_without 0.5 → 0.6, hard-impossible 조건 강화) |
| Stratification imbalance (특정 question type만 drop) | Filter rule 재고 |

### Gate 4 (Day 7 EOD) — Mini-GRPO training 후

읽을 file: 학습된 모델의 3 hard bench eval

| 조건 | 다음 단계 |
|---|---|
| R4 > R1 ≥ +2.5pp AND R4 > R2 ≥ +1.0pp | Full launch (5K sample, 7-day training) |
| R3/R4 ≈ R2 (modulation 효과 없음) | OC-VDM 폐기, Math-Shepherd 단독 paper narrative |
| R4 < R1 | 학습 자체 broken, 학습 spec audit |

---

## 5. Failure mode patches

### F1: Sub-rollout judge가 빈 pred → "Yes" 반환

이전 발견된 bug. **이미 fix 적용됨** (`if not pred.strip(): return 0`). Tier 1-B에서도 동일 보장 — 새 스크립트에 inline 적용 확인.

### F2: Image-hidden mode (mc_without) generation 실패

`image_dep_mc.py` 일부 sample에서 image 가린 user message 받았을 때 generation 실패 가능.
- Mitigation: text-only message에 explicit "[Chart image hidden]" 표시 + question + previous steps prefix
- Verify: 50 sample dry-run으로 빈 응답 비율 측정

### F3: Filter가 특정 bench를 거의 다 drop

예: chartmuseum에서 80% 이상이 hard-impossible로 분류.
- Threshold 완화: `mc_w_max == 0 AND mc_wo_max == 0` → `mc_w_max < 0.05`
- 또는 filter 부분 적용: chartmuseum만 filter, 다른 bench는 그대로

### F4: Mini-GRPO 학습 중 reward variance 0 dominant

학습 중 group 내 모든 trajectory가 같은 outcome (모두 fail or 모두 success).
- K_outer 증가 (8 → 16)
- Sample 난이도 stratified balance (filter 통과한 sample 중 mc_w 0.2-0.8 sample만 사용)

### F5: OC-VDM advantage이 chart-specific signal과 충돌

만약 vd 양수임에도 학습 안 좋아지면:
- λ 감소 (1.0 → 0.5)
- Modulation floor 상승 (0.2 → 0.5)
- 또는 modulation off (pure MS-MC로 fallback)

---

## 6. Cost / Timeline summary

| Phase | Day | GPU-hour | API | Manual |
|---|---|---|---|---|
| Tier 1-B measurement (12 combo) | D1 | 2-3h | $0 | 0 |
| Boundary sensitivity ablation | D1 EOD | 30min | $0 | 0 |
| Advantage simulation + verbatim | D2 | 0 (CPU) | $0 | 30min (verify table) |
| Filter stratification | D2 EOD | 0 (CPU) | $0 | 10min (report review) |
| Mini-GRPO (조건부) | D3-D6 | ~24h | $0 | 0 |
| Eval on 3 hard benches | D7 | ~6h | $0 | 30min |
| **Total to GO/NO-GO** | **7 days** | **~36h** | **$0** | **~70min** |

---

## 7. 산출물 list (research agent → 멘토 보고)

D1 EOD:
- `data/d2_hardbench/reports/image_dep_12combo/*.jsonl` (12 files)
- `data/d2_hardbench/reports/oc_vdm_cross_bench.json`
- `data/d2_hardbench/reports/boundary_sensitivity.json`
- Interim: H1, H2 결과 + Gate 1 decision

D2 EOD:
- `data/d2_hardbench/reports/oc_vdm_advantage_sim_results.json`
- `data/d2_hardbench/reports/oc_vdm_filter_stats.json`
- Interim: H3, H4 결과 + Gate 2/3 decision

D7 EOD (조건부):
- `ckpt/oc_vdm_mini/R1`, `R2`, `R3`, `R4` 학습 체크포인트
- `data/d2_hardbench/eval/oc_vdm_mini_results.json` (3 bench × 4 design)
- `docs/oc_vdm_validation_report.md` (최종 결정 — full launch / pivot / drop)

---

## 8. 즉시 시작 체크리스트

연구 agent가 본 가이드 받으면:

1. [ ] `scripts/image_dep_mc_v2.py` 작성 (§3.2)
2. [ ] vLLM 서버 launch 확인:
   - 8501 GPU11 qwen3vl-4b-instruct (policy)
   - 10.1.211.147:8000 397B (judge)
3. [ ] 12 combo measurement launch (§2.1)
4. [ ] D1 EOD `oc_vdm_analyze.py` 실행, Gate 1 결정 → 멘토 보고
5. [ ] D2 advantage simulation + filter (§2.3, §2.4), Gate 2/3 결정
6. [ ] (Gate 3 PASS 시) D3-D7 mini-GRPO + eval (§2.5)

각 단계 EOD마다 결과 file + Gate decision을 멘토에게 보고. **새 design proposal은 멘토와 명시적 합의 후에만**.

---

## 9. 약속

본 가이드는 다음 원칙으로 작성됨:
- **Step content parsing 0** (chartmuseum_759 같은 free-form trace에 robust)
- **Outcome-based signal only** (claim type 분류, entity 추출, value verify 없음)
- **chartmuseum 11-sample 외 검증** (12 combo로 universal 확인)
- **Decision-gated** (각 Gate 통과 후에만 다음 phase)
- **모든 failure mode에 explicit fallback** (Math-Shepherd-only로 narrow 가능)

**연구 agent → 멘토**: design 변경 / 새 axis 제안 / scope 확장 필요 시 explicit user 허가 받음. 본 가이드 외 추가 작업 자체 결정 X.

---

_End of action guide._
