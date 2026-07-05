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


## Overview
Sensor Grounding Essays is a framework that converts physical sensor measurements 
into natural language reports (essays) using a deterministic pipeline that 
bypasses LLM distribution sampling.

The framework demonstrates a form of "causal grounding" for robotic communication: 
physical events (sensor measurements) are linked through explicit causal chains to natural language utterances and robotic actions.

## Components

- `gpu_telemetry.py`: ROCm telemetry extraction + SensorReading/Threshold abstractions
- `grounding_pipeline01.py`: 4/10 sentence generators, 5-sentence paragraph, 3-paragraph essay, timeseries multi-domain
- `build_grounding_dataset.py`: Auto-builds grounding dataset (9 essays)
- `grounding_dataset.json`: 9 essays (Battery 4 + Motor 4 + Multidomain 1)


## Key Features

- 4 sentence types (conditional/imperative/declarative/concessive)
- 10 complex sentence types (compound/causal/temporal/concessive/purpose/result/comparative/relative/concessive-adversative)
- 5-sentence paragraph (compose_paragraph_universal)
- 3-paragraph essay (compose_essay_universal)
- Timeseries + multi-domain essay (compose_essay_timeseries_multidomain)
- Comparison direction correctness (4 cases × 3 domains = 12 combinations verified)

## Use Cases

- Robot-robot communication (Robot Language)
- Robot-human communication (Robot-Human Common Language)
- Sensor-grounded reports for industrial robotics
- Grounding dataset for LLM evaluation

## Quick Start

```bash
python3 build_grounding_dataset.py

This generates grounding_dataset.json with 9 essays.

License
MIT License
