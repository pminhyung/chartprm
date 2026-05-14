# SFT lazy image loading — 2026-04-10

## Context / 배경

`train_sft.py::load_sft_dataset()` 는 원래 전체 학습 데이터 (43,242 samples for
v_hq-lite) 의 이미지를 **startup 시점에 eager load** 했다. 구체적으로:

1. `json.loads` 로 jsonl 전체 로드
2. 매 row 마다 `PIL.Image.open(path).convert("RGB") → resize` 를 즉시 실행
3. PIL 객체를 리스트에 쌓아 `Dataset.from_list(samples)` 로 변환

Accelerate + DeepSpeed ZeRO-3 로 `--num_processes 8` 실행하면 **8 worker
processes 가 각자 독립적으로** 43K PIL 로드를 수행해서 다음 현상이 발생했다:

- 2026-04-10 12:38 런에서 **48 분 경과 + GPU 0-7 는 3 MiB 에 머무름** (모델 로드
  까지 도달 못 함)
- 각 worker RSS 가 9-10 GB 까지 올라갔다 (PIL 객체 8× 중복)
- "Loaded X/43744" 프린트가 5000 단위 찍혀야 하는데 거의 찍히지 않음 → I/O 병목

추정: 43K × 8 worker × ~100-200 ms/image ≈ 2-5 시간 startup.

## Fix

`load_sft_dataset` 을 **lazy-transform 패턴**으로 변경.

1. JSONL scan 단계에서는 `image_path` 가 실제 존재하는지만 확인하고,
   row 에는 **경로 문자열** 만 저장 (`"image_path": img_path`).
2. `Dataset.from_list(samples)` 로 Dataset 생성 직후
   `ds.set_transform(_lazy_load)` 등록.
3. `_lazy_load(batch)` 는 DataLoader 가 배치를 꺼낼 때만 호출되며, 해당 배치에
   속한 경로들만 PIL open + resize + `batch["images"] = [[img]...]` 로 주입.
4. 읽기 실패 시 64×64 화이트 PIL 로 fallback (이미 scan 시 존재 확인했으므로
   현실적으로는 발생 안 함).

## Measured improvement (동일 데이터셋, 동일 하드웨어)

| 단계 | Eager (기존) | Lazy (수정 후) |
|---|---|---|
| JSONL 스캔 → Dataset 생성 | **≥ 48 분 (미완)** | ~**2 분** |
| Worker RSS @ scan 끝 | ~10 GB (PIL 객체) | ~1 GB (경로 문자열) |
| 모델 weights 로드 시작까지 | 측정 불가 (미도달) | 스캔 직후 (~3 초) |
| 723 weight shards 로드 | — | ~2.5 초 |

즉 startup 시간이 **30배 이상 단축**되었다. GPU 0-7 가 lazy 로 패치한 후에는
scan 종료 직후 곧바로 shard 로드 단계로 진입해 GPU 메모리에 샤드가 할당되기
시작했다. (2026-04-10 12:38 재런에서 scan 완료 2분 → LoRA config → SFTTrainer
init → GPU 0,2,4,6,7 에 546 MiB rank0 shard 할당 확인.)

## Snippet (현재 반영된 형태)

`train_sft.py::load_sft_dataset`:

```python
def load_sft_dataset(data_path: str, max_image_side: int = 1024) -> Dataset:
    """Load SFT data with LAZY image decoding.

    Previously this eagerly ran PIL.Image.open + resize on every sample up
    front and stored the PIL objects in the Dataset. With 43K samples × 8
    parallel worker processes (DeepSpeed) that was a >2-hour startup
    bottleneck.
    """
    with open(data_path) as f:
        raw = [json.loads(line) for line in f]

    samples = []
    skipped = 0
    for i, item in enumerate(raw):
        q = item["question"]
        ans = str(item["answer"])
        img_path = item.get("image_path", "")

        if not img_path or not os.path.exists(img_path):
            skipped += 1
            continue

        # <think>/<answer> 포맷 assembly (teacher assistant_text 우선,
        # 아니면 reasoning_steps 를 <think> 로 감싸는 기존 로직 그대로)
        assistant_text = item.get("assistant_text", "")
        if assistant_text:
            response = assistant_text
        else:
            reasoning = item.get("reasoning", item.get("reasoning_steps", ""))
            if isinstance(reasoning, list):
                reasoning = "\n".join(reasoning)
            if reasoning:
                response = f"<think>\n{reasoning}\n</think>\n<answer>{ans}</answer>"
            else:
                response = (
                    f"<think>\nLet me analyze the chart to answer this question.\n"
                    f"The answer is {ans}.\n</think>\n<answer>{ans}</answer>"
                )

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Look at this chart and answer the question.\n\nQuestion: {q}"},
            {"role": "assistant", "content": response},
        ]

        samples.append({
            "messages": messages,
            "image_path": img_path,   # ← 경로 문자열만 저장
        })

        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1}/{len(raw)} (skipped {skipped})")

    print(f"  SFT dataset: {len(samples)} rows scanned, {skipped} skipped from {data_path}")
    ds = Dataset.from_list(samples)

    def _lazy_load(batch):
        imgs = []
        for p in batch["image_path"]:
            try:
                img = Image.open(p).convert("RGB")
                m = max(img.size)
                if m > max_image_side:
                    s = max_image_side / m
                    img = img.resize(
                        (int(img.size[0] * s), int(img.size[1] * s)),
                        Image.LANCZOS,
                    )
            except Exception:
                img = Image.new("RGB", (64, 64), color="white")
            imgs.append([img])
        batch["images"] = imgs
        return batch

    ds.set_transform(_lazy_load)
    return ds
```

## 사용 시 주의사항

- `ds.set_transform` 은 **in-place** 로 동작하며 원본 row 에 `images` key 가
  없어도 DataLoader 가 각 배치를 요청할 때 추가해 준다.
- `SFTTrainer` 기본 collator (TRL ≥ 0.29) 는 `messages` + `images` 필드를 이미
  지원한다. 별도 custom collator 필요 없음.
- `num_workers` DataLoader 를 쓰면 worker process 가 PIL 로드를 병렬화해서
  GPU idle 도 줄일 수 있다. 현재 TRL SFTConfig 는 기본 0 이므로 필요하면
  `dataloader_num_workers=4` 정도로 bump 고려.
- set_transform 결과는 캐시되지 않는다 (매 epoch 마다 다시 로드). 1-epoch SFT
  라 상관없지만 multi-epoch 시에는 disk hot-cache 에 의존한다.

## Command recipe (재현용)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup \
  /ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate launch \
    --num_processes 8 \
    --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_sft.py \
    --model_path /ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b \
    --data data/sft_hq_lite_clean.jsonl \
    --output_dir ckpt/sft_hq_lite_clean_4b_8gpu \
    --use_lora --lora_rank 64 --lora_alpha 128 --max_length 4096 \
  </dev/null > /tmp/train_logs/sft_hq_lite_clean.log 2>&1 & disown
```

스캔 완료 시그널:

```
  Scanned 5000/43242 (skipped 0)
  ...
  Scanned 40000/43242 (skipped 0)
  SFT dataset: 43242 rows scanned, 0 skipped from data/sft_hq_lite_clean.jsonl
```

이후 `Loading weights: 100%`, `LoRA: rank=64, alpha=128` 가 순차적으로 나오면
정상 진행. GPU 0-7 에 수백 MB 메모리 할당 시작하면 SFTTrainer / deepspeed 가
shard 분배 중이라는 뜻.
