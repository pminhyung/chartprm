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

- GPUs 0-1: vLLM 생성 서버
- GPUs 2-11: GRPO 학습 (DeepSpeed ZeRO-3)
- **GPUs 12-15: 사용 금지** (다른 사용자)
