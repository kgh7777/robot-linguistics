"""
build_extended_dataset.py

지금까지 라이브로 검증된 essay들을 확장하여 그라운딩 데이터셋을 빌드.
- 4도메인 (GPU, Battery, Motor, Mixed)
- 각 도메인마다 8가지 (2 value x 2 comparison x 2 action)
- 시계열 + 다중 도메인 essay 추가
- 메타데이터 + 입력 벡터 + 출력 essay

총 essay 수: 4도메인 x 8 = 32 (단일 시점) + 4 (시계열) = 36개
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


def make_essay(domain, value, comparison, threshold_value, action_text, sensor_id, sensor_type='temperature', unit='C'):
    """단일 essay 생성."""
    readings = [gt.GpuSample(0, sensor_id, 'mock', temperature_c=value, power_w=0.0)]
    thresholds = [gt.Threshold(sensor_type, comparison, threshold_value, action_text)]

    essay = gp.compose_essay_universal(readings, thresholds)

    return {
        'domain': domain,
        'timesteps': 1,
        'value': value,
        'sensor_id': sensor_id,
        'sensor_type': sensor_type,
        'comparison': comparison,
        'threshold_value': threshold_value,
        'is_action': value >= threshold_value if comparison == '>=' else value <= threshold_value,
        'action_text': action_text,
        'inputs': [{'sensor_id': sensor_id, 'sensor_type': sensor_type, 'value': value, 'unit': unit}],
        'thresholds': [{'sensor_type': sensor_type, 'comparison': comparison, 'threshold_value': threshold_value, 'action_text': action_text}],
        'output_essay': essay,
        'timestamp': time.time(),
    }


def build_gpu_dataset():
    """GPU essay 8가지 조합."""
    cases = [
        (38.0, '<', 40.0, '', 'card0', False),  # 38 < 40 (False)
        (42.0, '<', 40.0, '', 'card0', True),   # 42 > 40 (True, action 발화)

        (38.0, '>', 40.0, '', 'card0', False),  # 38 < 40 (False)
        (42.0, '>', 40.0, '', 'card0', True),   # 42 > 40 (True, action 발화)

        (40.0, '<=', 40.0, 'Move quickly', 'card0', True),    # 40 <= 40 (boundary)
        (40.0, '>=', 40.0, 'Move quickly', 'card0', True),    # 40 >= 40 (boundary)
        (39.9, '<=', 40.0, 'Move quickly', 'card0', True),    # 39.9 <= 40 (boundary)
        (39.9, '>=', 40.0, 'Move quickly', 'card0', False),   # 39.9 < 40 (False)
    ]

    return [make_essay('gpu', v, c, t, a, sid) for v, c, t, a, sid, _ in cases]


def build_battery_dataset():
    """Battery essay 8가지 조합."""
    cases = [
        (12.4, '<', 11.0, '', 'battery1', False),
        (10.5, '<', 11.0, '', 'battery1', True),

        (12.4, '>', 11.0, '', 'battery1', True),
        (10.5, '>', 11.0, '', 'battery1', False),

        (11.0, '<=', 11.0, 'Return to base', 'battery1', True),
        (11.0, '>=', 11.0, 'Return to base', 'battery1', True),
        (11.1, '<=', 11.0, 'Return to base', 'battery1', False),
        (10.9, '>=', 11.0, 'Return to base', 'battery1', False),
    ]

    return [make_essay('battery', v, c, t, a, sid) for v, c, t, a, sid, _ in cases]


def build_motor_dataset():
    """Motor essay 8가지 조합."""
    cases = [
        (35.0, '<', 80.0, '', 'motor2', True),
        (85.0, '<', 80.0, '', 'motor2', False),

        (35.0, '>', 80.0, '', 'motor2', False),
        (85.0, '>', 80.0, '', 'motor2', True),

        (80.0, '<=', 80.0, 'Stop motor', 'motor2', True),
        (80.0, '>=', 80.0, 'Stop motor', 'motor2', True),
        (79.9, '<=', 80.0, 'Stop motor', 'motor2', True),
        (79.9, '>=', 80.0, 'Stop motor', 'motor2', False),
    ]

    return [make_essay('motor', v, c, t, a, sid) for v, c, t, a, sid, _ in cases]


def build_lidar_dataset():
    """LiDAR essay 8가지 조합."""
    cases = [
        (2.4, '<', 0.5, '', 'lidar_front', False),
        (0.3, '<', 0.5, '', 'lidar_front', True),

        (2.4, '>', 0.5, '', 'lidar_front', True),
        (0.3, '>', 0.5, '', 'lidar_front', False),

        (0.5, '<=', 0.5, 'Stop immediately', 'lidar_front', True),
        (0.5, '>=', 0.5, 'Stop immediately', 'lidar_front', True),
        (0.6, '<=', 0.5, 'Stop immediately', 'lidar_front', False),
        (0.4, '>=', 0.5, 'Stop immediately', 'lidar_front', False),
    ]

    return [make_essay('lidar', v, c, t, a, sid) for v, c, t, a, sid, _ in cases]


def build_multidomain_timeseries_dataset():
    """시계열 + 다중 도메인 essay (3가지 시나리오)."""
    scenarios = []

    # 시나리오 1: GPU rising, Battery stable, Motor stable
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

    scenarios.append({
        'domain': 'multidomain',
        'timesteps': 3,
        'scenario': 'gpu_rising_battery_motor_stable',
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

    # 시나리오 2: GPU falling, Battery action, Motor action
    gpu_history2 = [
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=55.0, power_w=30.0)],
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=50.0, power_w=25.0)],
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=45.0, power_w=20.0)],
    ]
    battery_history2 = [
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.0, power_w=0.0)],
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=11.5, power_w=0.0)],
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=10.8, power_w=0.0)],
    ]
    motor_history2 = [
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=70.0, power_w=0.0)],
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=75.0, power_w=0.0)],
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=85.0, power_w=0.0)],
    ]

    history2 = {'gpu': gpu_history2, 'battery': battery_history2, 'motor': motor_history2}

    essay2 = gp.compose_essay_timeseries_multidomain(history2, thresholds)

    scenarios.append({
        'domain': 'multidomain',
        'timesteps': 3,
        'scenario': 'gpu_falling_battery_motor_action',
        'domains': ['gpu', 'battery', 'motor'],
        'trends': {'gpu': 'falling', 'battery': 'falling', 'motor': 'rising'},
        'domain_status': {'gpu': 'stable', 'battery': 'action', 'motor': 'action'},
        'inputs': {
            'gpu': [[{'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in gpu_history2],
            'battery': [[{'sensor_id': 'battery1', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in battery_history2],
            'motor': [[{'sensor_id': 'motor2', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in motor_history2],
        },
        'thresholds': {
            'gpu': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': 40.0, 'action_text': 'Move quickly'}],
            'battery': [{'sensor_type': 'temperature', 'comparison': '<=', 'threshold_value': 11.0, 'action_text': 'Return to base'}],
            'motor': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': 80.0, 'action_text': 'Stop motor'}],
        },
        'output_essay': essay2,
        'timestamp': time.time(),
    })

    # 시나리오 3: All stable
    gpu_history3 = [
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=35.0, power_w=15.0)],
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=36.0, power_w=16.0)],
        [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=35.5, power_w=15.5)],
    ]
    battery_history3 = [
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.4, power_w=0.0)],
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.5, power_w=0.0)],
        [gt.GpuSample(0, 'battery1', 'mock', temperature_c=12.4, power_w=0.0)],
    ]
    motor_history3 = [
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=40.0, power_w=0.0)],
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=42.0, power_w=0.0)],
        [gt.GpuSample(0, 'motor2', 'mock', temperature_c=41.0, power_w=0.0)],
    ]

    history3 = {'gpu': gpu_history3, 'battery': battery_history3, 'motor': motor_history3}

    essay3 = gp.compose_essay_timeseries_multidomain(history3, thresholds)

    scenarios.append({
        'domain': 'multidomain',
        'timesteps': 3,
        'scenario': 'all_stable',
        'domains': ['gpu', 'battery', 'motor'],
        'trends': {'gpu': 'stable', 'battery': 'stable', 'motor': 'stable'},
        'domain_status': {'gpu': 'stable', 'battery': 'stable', 'motor': 'stable'},
        'inputs': {
            'gpu': [[{'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in gpu_history3],
            'battery': [[{'sensor_id': 'battery1', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in battery_history3],
            'motor': [[{'sensor_id': 'motor2', 'sensor_type': 'temperature', 'value': s.temperature_c} for s in t] for t in motor_history3],
        },
        'thresholds': {
            'gpu': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': 40.0, 'action_text': 'Move quickly'}],
            'battery': [{'sensor_type': 'temperature', 'comparison': '<=', 'threshold_value': 11.0, 'action_text': 'Return to base'}],
            'motor': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': 80.0, 'action_text': 'Stop motor'}],
        },
        'output_essay': essay3,
        'timestamp': time.time(),
    })

    return scenarios


def main():
    print("Building extended grounding dataset...")

    gpu = build_gpu_dataset()
    print(f"  GPU: {len(gpu)} essays")

    battery = build_battery_dataset()
    print(f"  Battery: {len(battery)} essays")

    motor = build_motor_dataset()
    print(f"  Motor: {len(motor)} essays")

    lidar = build_lidar_dataset()
    print(f"  LiDAR: {len(lidar)} essays")

    multidomain = build_multidomain_timeseries_dataset()
    print(f"  Multidomain: {len(multidomain)} essays")

    all_essays = gpu + battery + motor + lidar + multidomain

    output_path = '/home/kgh/grounding_dataset_extended.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_essays, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_essays)} essays to {output_path}")
    print(f"  By domain:")
    by_domain = {}
    for e in all_essays:
        d = e['domain']
        by_domain[d] = by_domain.get(d, 0) + 1
    for d, count in sorted(by_domain.items()):
        print(f"    {d}: {count}")


if __name__ == '__main__':
    main()
