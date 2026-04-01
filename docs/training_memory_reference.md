# Training Memory Reference

GPU: A100-40GB. 모든 테스트: LoRA r=64, alpha=128, DeepSpeed ZeRO-3 (no CPU offloading).

## Qwen3.5-4B + LoRA

### 필수 설정
```python
gradient_checkpointing=True
gradient_checkpointing_kwargs={"use_reentrant": True}  # Qwen3.5 hybrid attn 호환
attn_implementation="flash_attention_2"
# use_liger_kernel=True  ← Qwen3.5 미지원 (무시됨)
```

### 단일 GPU 메모리 (dry run, 2 steps)

| num_gen | max_comp | Peak Memory | Status |
|---------|----------|-------------|--------|
| 4       | 1024     | 13.0 GB     | ✅ OK  |
| 8       | 2048     | 18.0 GB     | ✅ OK  |
| **16**  | **4096** | **26.0 GB** | **✅ OK (권장)** |
| 16      | 8192     | >40 GB      | ❌ OOM |

### grad_ckpt OFF (이전 v7 Row A/B 설정, 참고용)

| num_gen | max_comp | Peak Memory | Status |
|---------|----------|-------------|--------|
| 4       | 1024     | 40.0 GB     | ✅ 빡빡 (ZeRO-3+CPU offload 필요) |
| 8       | 2048     | >40 GB      | ❌ OOM |

### 8 GPU 학습 추천 설정

```python
GRPOConfig(
    # DAPO
    loss_type="dapo",
    epsilon=0.2, epsilon_high=0.28, beta=0.0,
    mask_truncated_completions=True,
    # Generation
    num_generations=16,
    max_completion_length=4096,
    generation_batch_size=16,
    # Training
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # effective batch = 8×1×4 = 32
    learning_rate=1e-5,
    # Memory
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    bf16=True,
    max_grad_norm=1.0,
    # vLLM
    use_vllm=True, vllm_mode="server",
)

LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj",
                           "gate_proj","up_proj","down_proj"])
```

**GPU 배분**: 1 GPU (vLLM server) + 8 GPU (training) = 9 GPU 필요
**예상 메모리**: ~26 GB/GPU (여유 14 GB)
**예상 시간**: ~6-8h for 3916 samples

## Qwen3.5-9B + LoRA

### 단일 GPU 메모리 (dry run, 2 steps)

| num_gen | max_comp | Peak Memory | Status |
|---------|----------|-------------|--------|
| 4       | 1024     | 23.2 GB     | ✅ OK  |
| 8       | 2048     | 27.3 GB     | ✅ OK  |
| 16      | 2048     | 27.4 GB     | ✅ OK  |
| 16      | 4096     | >34.6 GB    | ❌ OOM (1 GPU) |

### 8 GPU ZeRO-3 (no CPU offload)

| num_gen | max_comp | Per-GPU Memory | Status |
|---------|----------|----------------|--------|
| **16**  | **4096** | **23.0 GB**    | **✅ OK (권장)** |

**주의**: 1 GPU에서 OOM이지만 8 GPU ZeRO-3에서는 23GB로 동작 (모델 파라미터 분산 효과).

### 8 GPU 학습 추천 설정

```python
# 9B는 vLLM TP=1 generation이 느림. 학습 GPU 8개 + vLLM 1개 = 9 GPU 필요.
# vLLM generation이 bottleneck이므로 num_gen을 줄이는 것도 고려.
GRPOConfig(
    num_generations=16,       # 또는 8 (generation 속도 vs variance trade-off)
    max_completion_length=4096,
    # 나머지는 4B와 동일
)
```

**vLLM 참고**: 9B TP=1 서버는 16×4096 토큰 generation에 ~10분+ 소요.
TP=2로 올리면 속도 ~2배지만 GPU 1개 추가 필요.

## 주의사항

- **Liger Kernel**: Qwen3.5 모델 타입 미지원 (`"no Liger kernels supported for model type: qwen3_5"`)
- **gradient_checkpointing**: 반드시 `use_reentrant=True` 사용. `False`나 미지정 시 `CheckpointError`
- **CPU offloading**: 가능하면 사용 안 함 (속도 ~25% 저하). grad_ckpt로 대부분 해결됨
- **batch size 통제**: 실험 간 `effective_batch = num_gpus × per_device_batch × grad_accum` 동일하게 유지
