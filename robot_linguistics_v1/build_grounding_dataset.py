"""
build_grounding_dataset.py

지금까지 라이브로 검증된 essay들을 자동으로 그라운딩 데이터셋으로 빌드.

각 essay에 메타데이터(domain, value, comparison, is_action, timestamp)를 추가하고
JSON 파일로 저장.
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


def build_battery_dataset():
    """Battery essay 4가지 조합."""
    cases = [
        (12.4, '<=', 11.0, 'Return to base', False),
        (10.5, '<=', 11.0, 'Return to base', True),
        (12.4, '>=', 11.0, 'Alert', True),
        (10.5, '>=', 11.0, 'Alert', False),
    ]

    dataset = []
    for value, comparison, threshold_value, action_text, is_action in cases:
        readings = [gt.GpuSample(0, 'battery1', 'mock', temperature_c=value, power_w=0.0)]
        thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]

        essay = gp.compose_essay_universal(readings, thresholds)

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

    return dataset


def build_motor_dataset():
    """Motor essay 4가지 조합."""
    cases = [
        (35.0, '>=', 80.0, 'Stop motor', False),
        (85.0, '>=', 80.0, 'Stop motor', True),
        (35.0, '<=', 20.0, 'Heat motor', False),
        (15.0, '<=', 20.0, 'Heat motor', True),
    ]

    dataset = []
    for value, comparison, threshold_value, action_text, is_action in cases:
        readings = [gt.GpuSample(0, 'motor2', 'mock', temperature_c=value, power_w=0.0)]
        thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]

        essay = gp.compose_essay_universal(readings, thresholds)

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

    return dataset


def build_multidomain_timeseries_dataset():
    """시계열 + 다중 도메인 essay (방금 라이브 검증)."""
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

    essay = gp.compose_essay_timeseries_multidomain(history, thresholds)

    return [{
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
    }]


def main():
    print("Building grounding dataset...")

    battery = build_battery_dataset()
    print(f"  Battery: {len(battery)} essays")

    motor = build_motor_dataset()
    print(f"  Motor: {len(motor)} essays")

    multidomain = build_multidomain_timeseries_dataset()
    print(f"  Multidomain: {len(multidomain)} essays")

    all_essays = battery + motor + multidomain

    output_path = '/home/kgh/grounding_dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_essays, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_essays)} essays to {output_path}")


if __name__ == '__main__':
    main()
