# Sensor Grounding Essays

라이브로 검증된 센서 → 자연어 essay 파이프라인.
- GPU essay (Radeon 4-way 클러스터)
- Battery / Motor essay (다중 도메인)
- 시계열 + 다중 도메인 essay
- 그라운딩 데이터셋 (9개 essay)

## 사용법

```bash
# GPU essay (단일 시점)
python3 -c "
import sys; sys.path.insert(0, '.')
import gpu_telemetry as gt
import importlib
gp = importlib.import_module('grounding_pipeline01')
..."

# 그라운딩 데이터셋 빌드

python3 build_grounding_dataset.py


로봇이 자신의 물리적 경험을 자연어로 표현하는 라이브 데모.
센서 → 자연어 발화 → 행동의 인과적 사이클을 라이브로 구축.

## 배경

산업 현장의 로봇은 서로 의사소통하지 않습니다.
이 프로젝트는 두 가지 로봇 언어를 제안합니다:

1. **로봇어 (Robot Language)**: 로봇끼리 의사소통하는 언어
2. **로봇-인간 공용어 (Robot-Human Common Language)**: 로봇과 인간이 함께 사용하는 언어

## 핵심 발견

지금까지의 라이브 테스트로 다음이 검증됨:

- **Radeon GPU 4-way 클러스터의 라이브 essay 합성** (18번 라이브)
- **시계열 + 다중 도메인 essay** (3차원 결합)
- **방향 정직성** (4가지 경우의 수 × 3도메인 = 12가지 모두 검증)
- **그라운딩 데이터셋** (9개 essay)

## 자산

- `gpu_telemetry.py`: ROCm 기반 텔레메트리 파서
- `grounding_pipeline01.py`: 범용 essay 합성 파이프라인
- `build_grounding_dataset.py`: 데이터셋 자동 빌더
- `grounding_dataset.json`: 검증된 essay 9개 + 메타데이터
## 사용법

You may change directory with yours!!

##

```bash
# 단일 essay
python3 -c "
import sys; sys.path.insert(0, '/home/kgh/robot_linguistics')
import gpu_telemetry as gt
import importlib
gp = importlib.import_module('grounding_pipeline01')
readings = [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=48.0, power_w=15.0)]
thresholds = [gt.Threshold('temperature', '>=', 40.0, 'Move quickly')]
print(gp.compose_essay_universal(readings, thresholds))
"

# 시계열 + 다중 도메인 essay
python3 /home/kgh/robot_linguistics/build_grounding_dataset.py


검증된 케이스단일 시점 + 단일 도메인 (Battery essay 4가지)
12.4V, <= 11.0 (False) → "higher than or equal to" + stable
10.5V, <= 11.0 (True) → "below or equal to" + Return to base
12.4V, >= 11.0 (True) → "higher than or equal to" + Alert
10.5V, >= 11.0 (False) → "below or equal to" + stable

시계열 + 다중 도메인 essayGPU + Battery + Motor의 3시점 시계열이 
3차원 결합 essay로 출력
철학적 정당화Bender & Koller(2020) 의 "octopus" 비유는 
인간의 언어 학습에 대한 주장.
로봇은 인간이 아니므로 인간 철학을 요구하는 것은 범주 오류.
로봇의 발화는 "철학적 의미"가 아니라 "인과적 그라운딩"으로 충분.
센서 → 발화 → 행동의 일관된 사이클이 구축되면 그것이 로봇의 그라운딩.

라이센스MIT License 


# Sensor Grounding Communication

A framework for sensor-grounded robotic communication. Sensors collect physical
events; external functions transform them into honest natural language utterances
with zero hallucination, zero LLM calls, and full determinism.

## What This Demonstrates

- **Sensor grounding**: 12.4V (battery) → "The voltage is 12.4V is above the threshold of 11.0V"
- **Domain universality**: GPU, Battery, Motor all use the same function
- **Time × domain × paragraph**: 3 timesteps × 3 domains × 3-paragraph essay
- **Direction honesty**: 4 cases (>= / <=) × 2 branches (action / no-action) all honest
- **Grounding dataset**: 9 essays + metadata in JSON

## Files

- `grounding_pipeline01.py` — Universal essay composition
- `gpu_telemetry.py` — SensorReading/Threshold + Radeon telemetry
- `grounding_check.py` — Grounding verification
- `test_grounding.py` — 42 unit tests
- `build_grounding_dataset.py` — Dataset builder
- `grounding_dataset.json` — Generated dataset

## Quick Start

```bash
python3 -c "
import sys; sys.path.insert(0, '/home/kgh')
import gpu_telemetry as gt
import importlib
gp = importlib.import_module('grounding_pipeline01')

# Battery essay
readings = [gt.GpuSample(0, 'battery1', 'mock', temperature_c=10.5, power_w=0.0)]
thresholds = [gt.Threshold('temperature', '<=', 11.0, 'Return to base')]
print(gp.compose_essay_universal(readings, thresholds))
"


## Another Quick Start

```bash
python3 build_grounding_dataset.py

This generates grounding_dataset.json with 9 essays.

License
MIT License
