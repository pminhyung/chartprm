# ChartVCR Project

Chart-Verifiable Causal Rewards for Chart Reasoning via GRPO.

## 핵심 가이드 문서

- **`docs/quickstart_guide.md`** — 학습/추론 최적 환경 및 즉시 실행 가이드. 환경, TRL 패치, GRPO 학습 커맨드, 하이퍼파라미터, OOM 방지 설정 포함.
- **`docs/experiment_log.md`** — 환경/DeepSpeed/GPU 조합별 시도 결과 전체 기록. OOM 케이스, 디버깅 히스토리, 좀비 프로세스 대응 포함.

## 환경

- **Python**: `/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python` (유일하게 동작하는 환경)
- torch 2.10.0+cu128, vLLM 0.17.1rc1, transformers 5.3.0.dev0, TRL 0.29.0 (패치됨)
- **TRL 재설치 금지** — 패치가 사라짐

## GPU 정책

- GPUs 2-11: 사용 가능 (GRPO 학습 + vLLM)
- GPUs 0-1, 12-15: 유저 허가 시에만 사용
- **절대 `nvidia-smi --gpu-reset` 실행 금지** — 다른 사용자 프로세스에 영향. zombie GPU memory 발생 시 유저에게 PID 기반 kill 커맨드를 제공할 것.
- **절대 `fuser -k /dev/nvidia*` 실행 금지**
- 프로세스 kill은 반드시 PID 기반으로만 (`kill <PID>`)
