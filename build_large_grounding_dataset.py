"""
build_large_grounding_dataset.py

방법 1, 2, 3을 조합해서 500-1000개 essay를 자동 생성.

방법 1: 랜덤 도메인 조합
방법 2: 랜덤 시나리오
방법 3: 4가지 경우의 수 × 다중 도메인 × 다중 시점
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import random
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')

# 도메인 정의
DOMAINS = {
    'gpu': {
        'sensor_type': 'temperature',
        'unit': 'C',
        'value_range': (30.0, 90.0),
        'threshold_range': (40.0, 70.0),
        'action_pool': ['Move quickly', 'Reduce load', 'Cool down'],
        'comparisons': ['>=', '<='],
    },
    'battery': {
        'sensor_type': 'voltage',
        'unit': 'V',
        'value_range': (8.0, 14.0),
        'threshold_range': (10.0, 12.0),
        'action_pool': ['Return to base', 'Recharge', 'Reduce load'],
        'comparisons': ['<=', '>='],
    },
    'motor': {
        'sensor_type': 'rpm',
        'unit': 'rpm',
        'value_range': (1000.0, 6000.0),
        'threshold_range': (3000.0, 5000.0),
        'action_pool': ['Reduce load', 'Stop motor', 'Cool down'],
        'comparisons': ['>=', '<='],
    },
    'lidar': {
        'sensor_type': 'distance',
        'unit': 'm',
        'value_range': (0.1, 10.0),
        'threshold_range': (0.5, 2.0),
        'action_pool': ['Stop immediately', 'Slow down', 'Turn left'],
        'comparisons': ['<=', '>='],
    },
    'camera': {
        'sensor_type': 'brightness',
        'unit': '/255',
        'value_range': (0, 255),
        'threshold_range': (30, 100),
        'action_pool': ['Increase light', 'Adjust camera', 'Reduce light'],
        'comparisons': ['<=', '>='],
    },
    'mic': {
        'sensor_type': 'db',
        'unit': 'dB',
        'value_range': (20, 120),
        'threshold_range': (60, 100),
        'action_pool': ['Sound alarm', 'Reduce noise', 'Mute'],
        'comparisons': ['>=', '<='],
    },
}


def generate_single_domain_cases(n_cases=200):
    """방법 1: 단일 도메인, 4가지 경우의 수 (랜덤)."""
    cases = []
    for _ in range(n_cases):
        domain = random.choice(list(DOMAINS.keys()))
        cfg = DOMAINS[domain]

        value = random.uniform(*cfg['value_range'])
        threshold_value = random.uniform(*cfg['threshold_range'])
        comparison = random.choice(cfg['comparisons'])
        action = random.choice(cfg['action_pool'])

        # 1개 reading
        reading = gt.GpuSample(
            0,
            f"{domain}_sensor",
            'mock',
            temperature_c=value if cfg['sensor_type'] == 'temperature' else None,
            power_w=0.0,
        )
        # 직접 SensorReading 형식으로
        # gpu_telemetry의 SensorReading 사용
        readings = [gt.SensorReading(
            sensor_id=f"{domain}_sensor",
            sensor_type=cfg['sensor_type'],
            value=value,
            unit=cfg['unit'],
            timestamp=time.time(),
        )]
        thresholds = [gt.Threshold(cfg['sensor_type'], comparison, threshold_value, action)]

        essay = gp.compose_essay_universal(readings, thresholds)

        # is_action 계산
        if comparison == '>=':
            is_action = value >= threshold_value
        else:
            is_action = value <= threshold_value

        cases.append({
            'domain': domain,
            'sensor_type': cfg['sensor_type'],
            'value': value,
            'unit': cfg['unit'],
            'comparison': comparison,
            'threshold_value': threshold_value,
            'is_action': is_action,
            'action_text': action if is_action else '',
            'inputs': [{'sensor_id': f"{domain}_sensor", 'sensor_type': cfg['sensor_type'], 'value': value, 'unit': cfg['unit']}],
            'thresholds': [{'sensor_type': cfg['sensor_type'], 'comparison': comparison, 'threshold_value': threshold_value, 'action_text': action}],
            'output_essay': essay,
            'timestamp': time.time(),
        })

    return cases


def generate_multi_domain_cases(n_cases=200):
    """방법 2: 다중 도메인, 단일 시점."""
    cases = []
    for _ in range(n_cases):
        n_domains = random.randint(2, 4)
        selected = random.sample(list(DOMAINS.keys()), n_domains)

        readings = []
        thresholds = []

        for domain in selected:
            cfg = DOMAINS[domain]
            value = random.uniform(*cfg['value_range'])
            threshold_value = random.uniform(*cfg['threshold_range'])
            comparison = random.choice(cfg['comparisons'])
            action = random.choice(cfg['action_pool'])

            readings.append(gt.SensorReading(
                sensor_id=f"{domain}_sensor",
                sensor_type=cfg['sensor_type'],
                value=value,
                unit=cfg['unit'],
                timestamp=time.time(),
            ))
            thresholds.append(gt.Threshold(cfg['sensor_type'], comparison, threshold_value, action))

        essay = gp.compose_essay_universal(readings, thresholds)

        cases.append({
            'domain': f"multi_{'+'.join(selected)}",
            'sensor_types': [DOMAINS[d]['sensor_type'] for d in selected],
            'values': [r.value for r in readings],
            'units': [r.unit for r in readings],
            'comparisons': [t.comparison for t in thresholds],
            'threshold_values': [t.threshold_value for t in thresholds],
            'inputs': [{'sensor_id': r.sensor_id, 'sensor_type': r.sensor_type, 'value': r.value, 'unit': r.unit} for r in readings],
            'thresholds': [{'sensor_type': t.sensor_type, 'comparison': t.comparison, 'threshold_value': t.threshold_value, 'action_text': t.action_text} for t in thresholds],
            'output_essay': essay,
            'timestamp': time.time(),
        })

    return cases


def generate_timeseries_cases(n_cases=200):
    """방법 3: 시계열 (3-5시점) × 다중 도메인."""
    cases = []
    for _ in range(n_cases):
        n_domains = random.randint(2, 3)
        n_timesteps = random.randint(3, 5)
        selected = random.sample(list(DOMAINS.keys()), n_domains)

        history = {}
        thresholds_dict = {}

        for domain in selected:
            cfg = DOMAINS[domain]
            threshold_value = random.uniform(*cfg['threshold_range'])
            comparison = random.choice(cfg['comparisons'])
            action = random.choice(cfg['action_pool'])

            # 시계열 값 생성
            base_value = random.uniform(*cfg['value_range'])
            timestep_values = []
            for t in range(n_timesteps):
                # 시간에 따라 약간 변화
                delta = random.uniform(-5.0, 5.0)
                timestep_values.append(base_value + delta * t / 2)

            timestep_readings = []
            for v in timestep_values:
                timestep_readings.append([gt.SensorReading(
                    sensor_id=f"{domain}_sensor",
                    sensor_type=cfg['sensor_type'],
                    value=v,
                    unit=cfg['unit'],
                    timestamp=time.time(),
                )])

            history[domain] = timestep_readings
            thresholds_dict[domain] = [gt.Threshold(cfg['sensor_type'], comparison, threshold_value, action)]

        essay = gp.compose_essay_timeseries_multidomain(history, thresholds_dict)

        cases.append({
            'domain': f"timeseries_{'+'.join(selected)}",
            'timesteps': n_timesteps,
            'domains': selected,
            'history': {d: [[{'sensor_id': r[0].sensor_id, 'sensor_type': r[0].sensor_type, 'value': r[0].value, 'unit': r[0].unit} for r in ts] for ts in history[d]] for d in selected},
            'thresholds': {d: [{'sensor_type': thresholds_dict[d][0].sensor_type, 'comparison': thresholds_dict[d][0].comparison, 'threshold_value': thresholds_dict[d][0].threshold_value, 'action_text': thresholds_dict[d][0].action_text}] for d in selected},
            'output_essay': essay,
            'timestamp': time.time(),
        })

    return cases


def main():
    random.seed(42)  # 재현 가능성

    print("Building large grounding dataset (방법 1, 2, 3 조합)...")

    # 방법 1: 200개 (단일 도메인)
    print("  방법 1: 단일 도메인 essay 생성 중...")
    single = generate_single_domain_cases(n_cases=200)
    print(f"    생성 완료: {len(single)} essays")

    # 방법 2: 200개 (다중 도메인, 단일 시점)
    print("  방법 2: 다중 도메인 essay 생성 중...")
    multi = generate_multi_domain_cases(n_cases=200)
    print(f"    생성 완료: {len(multi)} essays")

    # 방법 3: 200개 (시계열 × 다중 도메인)
    print("  방법 3: 시계열 + 다중 도메인 essay 생성 중...")
    timeseries = generate_timeseries_cases(n_cases=200)
    print(f"    생성 완료: {len(timeseries)} essays")

    all_essays = single + multi + timeseries

    output_path = '/home/kgh/grounding_dataset_large.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_essays, f, indent=2, ensure_ascii=False)

    print(f"\n총 {len(all_essays)} essays를 {output_path}에 저장")
    print(f"  방법 1 (단일 도메인): {len(single)}")
    print(f"  방법 2 (다중 도메인): {len(multi)}")
    print(f"  방법 3 (시계열): {len(timeseries)}")


if __name__ == '__main__':
    main()
