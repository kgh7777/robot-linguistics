"""
robot_linguistics.py

로봇 언어학의 첫 라이브 데모.

핵심 자산:
- SensorReading / Threshold: 도메인 무관 추상 데이터 클래스
- compose_paragraph_universal: 5문장 단락 (방향 정직성 보강)
- compose_essay_universal: 3-Paragraph essay
- compose_essay_timeseries_multidomain: 시계열 × 다중 도메인 essay
- build_grounding_dataset: 그라운딩 데이터셋 자동 빌더

기반:
- GPU essay (Radeon 4-way, 18 라이브)
- Battery essay (4가지 경우의 수)
- Motor essay (4가지 경우의 수)
- Multidomain essay (시계열 × 3도메인)
- Grounding dataset (9 essays)
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


# ============================================================
# 1) 4종 단일 시점 단락
# ============================================================

def paragraph_single_domain(readings, thresholds):
    """단일 시점, 단일 도메인 5문장 단락."""
    return gp.compose_paragraph_universal(readings, thresholds)


# ============================================================
# 2) 3단락 essay
# ============================================================

def essay_single_domain(readings, thresholds):
    """단일 시점, 단일 도메인 3-Paragraph essay."""
    return gp.compose_essay_universal(readings, thresholds)


# ============================================================
# 3) 시계열 × 다중 도메인 essay
# ============================================================

def essay_timeseries_multidomain(history_per_domain, thresholds_per_domain):
    """시계열 × 다중 도메인 essay (3차원 결합)."""
    return gp.compose_essay_timeseries_multidomain(
        history_per_domain, thresholds_per_domain
    )


# ============================================================
# 4) 그라운딩 데이터셋 자동 빌더
# ============================================================

def build_dataset():
    """
    지금까지 라이브로 검증된 essay들을 JSON으로 빌드.
    Battery 4 + Motor 4 + Multidomain 1 = 9 essays.
    """
    dataset = []

    # Battery 4 cases
    battery_cases = [
        (12.4, '<=', 11.0, 'Return to base', False),
        (10.5, '<=', 11.0, 'Return to base', True),
        (12.4, '>=', 11.0, 'Alert', True),
        (10.5, '>=', 11.0, 'Alert', False),
    ]
    for value, comparison, threshold_value, action_text, is_action in battery_cases:
        readings = [gt.GpuSample(0, 'battery1', 'mock', temperature_c=value, power_w=0.0)]
        thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]
        essay = essay_single_domain(readings, thresholds)
        dataset.append({
            'domain': 'battery',
            'timesteps': 1,
            'value': value,
            'comparison': comparison,
            'threshold_value': threshold_value,
            'is_action': is_action,
            'action_text': action_text,
            'inputs': [{'sensor_id': 'battery1', 'sensor_type': 'temperature', 'value': value, 'unit': 'C'}],
            'thresholds': [{'sensor_type': 'temperature', 'comparison': comparison, 'threshold_value': threshold_value, 'action_text': action_text}],
            'output_essay': essay,
            'timestamp': time.time(),
        })

    # Motor 4 cases
    motor_cases = [
        (35.0, '>=', 80.0, 'Stop motor', False),
        (85.0, '>=', 80.0, 'Stop motor', True),
        (35.0, '<=', 20.0, 'Heat motor', False),
        (15.0, '<=', 20.0, 'Heat motor', True),
    ]
    for value, comparison, threshold_value, action_text, is_action in motor_cases:
        readings = [gt.GpuSample(0, 'motor2', 'mock', temperature_c=value, power_w=0.0)]
        thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]
        essay = essay_single_domain(readings, thresholds)
        dataset.append({
            'domain': 'motor',
            'timesteps': 1,
            'value': value,
            'comparison': comparison,
            'threshold_value': threshold_value,
            'is_action': is_action,
            'action_text': action_text,
            'inputs': [{'sensor_id': 'motor2', 'sensor_type': 'temperature', 'value': value, 'unit': 'C'}],
            'thresholds': [{'sensor_type': 'temperature', 'comparison': comparison, 'threshold_value': threshold_value, 'action_text': action_text}],
            'output_essay': essay,
            'timestamp': time.time(),
        })

    # Multidomain 1 case
    gpu_history = [
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=44.0, power_w=20.0)],
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=46.0, power_w=22.0)],
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=48.0, power_w=25.0)],
    ]
    gpu_thresholds = [gt.Threshold('temperature', '>=', 40.0, 'Move quickly')]
    battery_history = [
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.5, power_w=0.0)],
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.3, power_w=0.0)],
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.4, power_w=0.0)],
    ]
    battery_thresholds = [gt.Threshold('temperature', '<=', 11.0, 'Return to base')]
    motor_history = [
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=30.0, power_w=0.0)],
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=32.0, power_w=0.0)],
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=35.0, power_w=0.0)],
    ]
    motor_thresholds = [gt.Threshold('temperature', '>=', 80.0, 'Stop motor')]

    history = {'gpu': gpu_history, 'battery': battery_history, 'motor': motor_history}
    thresholds = {'gpu': gpu_thresholds, 'battery': battery_thresholds, 'motor': motor_thresholds}
    essay = essay_timeseries_multidomain(history, thresholds)
    dataset.append({
        'domain': 'multidomain',
        'timesteps': 3,
        'domains': ['gpu', 'battery', 'motor'],
        'trends': {'gpu': 'rising', 'battery': 'stable', 'motor': 'rising'},
        'domain_status': {'gpu': 'action', 'battery': 'stable', 'motor': 'stable'},
        'inputs': {
            'gpu': [[{'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in gpu_history],
            'battery': [[{'sensor_id': 'battery1', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in battery_history],
            'motor': [[{'sensor_id': 'motor2', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in motor_history],
        },
        'thresholds': {
            'gpu': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': 40.0, 'action_text': 'Move quickly'}],
            'battery': [{'sensor_type': 'temperature', 'comparison': '<=', 'threshold_value': 11.0, 'action_text': 'Return to base'}],
            'motor': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': 80.0, 'action_text': 'Stop motor'}],
        },
        'output_essay': essay,
        'timestamp': time.time(),
    })

    return dataset


# ============================================================
# 5) 메인: 그라운딩 데이터셋 저장
# ============================================================

def main():
    print("=" * 60)
    print("Robot Linguistics v1 - Grounding Dataset Builder")
    print("=" * 60)

    dataset = build_dataset()
    print(f"Built {len(dataset)} essays:")
    print(f"  Battery: 4 essays")
    print(f"  Motor: 4 essays")
    print(f"  Multidomain: 1 essay")

    output_path = '/home/kgh/robot_linguistics_v1/grounding_dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
