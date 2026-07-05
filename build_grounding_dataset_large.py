"""
build_grounding_dataset_large.py

100-1000 essay 규모의 그라운딩 데이터셋 자동 빌드.

여러 도메인 (GPU, Battery, Motor) × 여러 시나리오 × 시계열 × 다중 카드
조합으로 대량의 essay를 생성.
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import random
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


def generate_random_scenario(domain='battery'):
    """
    랜덤 시나리오 생성.
    domain: 'battery', 'motor', 'gpu'
    """
    if domain == 'battery':
        sensor_id = 'battery1'
        sensor_type = 'voltage'
        value_range = (10.0, 13.0)
        threshold_range = (10.5, 12.5)
        comparisons = ['<=', '>=']
        actions = ['Return to base', 'Alert', 'Reduce load', 'Recharge']
    elif domain == 'motor':
        sensor_id = 'motor2'
        sensor_type = 'rpm'
        value_range = (1000, 6000)
        threshold_range = (2000, 5000)
        comparisons = ['>=', '<=']
        actions = ['Reduce load', 'Stop motor', 'Increase power', 'Cool motor']
    elif domain == 'gpu':
        sensor_id = 'card0'
        sensor_type = 'temperature'
        value_range = (30.0, 80.0)
        threshold_range = (40.0, 75.0)
        comparisons = ['>=', '<=']
        actions = ['Move quickly', 'Cool GPU', 'Reduce load', 'Stop processing']
    else:
        raise ValueError(f"Unknown domain: {domain}")

    value = round(random.uniform(*value_range), 1)
    threshold = round(random.uniform(*threshold_range), 1)
    comparison = random.choice(comparisons)
    action = random.choice(actions)

    return {
        'sensor_id': sensor_id,
        'sensor_type': sensor_type,
        'value': value,
        'unit': 'C' if sensor_type == 'temperature' else ('V' if sensor_type == 'voltage' else 'rpm'),
        'threshold': threshold,
        'comparison': comparison,
        'action': action,
    }


def build_single_essay_dataset(n_per_domain=100):
    """
    단일 시점 essay 데이터셋.
    n_per_domain: 도메인당 essay 수 (배터리 100 + 모터 100 + GPU 100 = 300)
    """
    dataset = []
    domains = ['battery', 'motor', 'gpu']

    for domain in domains:
        for _ in range(n_per_domain):
            scenario = generate_random_scenario(domain)

            # reading 생성
            readings = [gt.GpuSample(0, scenario['sensor_id'], 'mock',
                                     temperature_c=scenario['value'] if scenario['sensor_type'] == 'temperature' else 25.0,
                                     power_w=scenario['value'] if scenario['sensor_type'] != 'temperature' else 0.0)]

            # threshold 생성
            thresholds = [gt.Threshold(scenario['sensor_type'], scenario['comparison'],
                                      scenario['threshold'], scenario['action'])]

            # essay 생성
            essay = gp.compose_essay_universal(readings, thresholds)

            # 데이터셋 항목
            dataset.append({
                'domain': domain,
                'timesteps': 1,
                'value': scenario['value'],
                'comparison': scenario['comparison'],
                'threshold_value': scenario['threshold'],
                'action_text': scenario['action'],
                'inputs': [{'sensor_id': scenario['sensor_id'],
                            'sensor_type': scenario['sensor_type'],
                            'value': scenario['value'],
                            'unit': scenario['unit']}],
                'thresholds': [{'sensor_type': scenario['sensor_type'],
                                'comparison': scenario['comparison'],
                                'threshold_value': scenario['threshold'],
                                'action_text': scenario['action']}],
                'output_essay': essay,
                'timestamp': time.time(),
            })

    return dataset


def build_timeseries_dataset(n=100):
    """
    시계열 essay 데이터셋.
    n: 시계열 essay 수
    """
    dataset = []

    for i in range(n):
        # 3시점 시계열 (랜덤 추세)
        t1 = round(random.uniform(20.0, 50.0), 1)
        delta = random.uniform(-5.0, 10.0)
        t2 = round(t1 + delta / 2, 1)
        t3 = round(t1 + delta, 1)

        # 임계치 (랜덤)
        threshold = round(random.uniform(30.0, 60.0), 1)
        action = random.choice(['Move quickly', 'Reduce load', 'Cool GPU', 'Stop motor'])

        history = [
            [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=t1, power_w=20.0)],
            [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=t2, power_w=22.0)],
            [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=t3, power_w=25.0)],
        ]
        thresholds = [gt.Threshold('temperature', '>=', threshold, action)]

        # 가장 최근 측정값으로 단락 생성
        latest_readings = history[-1]

        # 단일 도메인 시계열 essay (compose_paragraph_universal 사용)
        body = gp.compose_paragraph_universal(latest_readings, thresholds)

        # 추세 분석
        delta_total = t3 - t1
        if delta_total > 1.0:
            trend = 'rising'
        elif delta_total < -1.0:
            trend = 'falling'
        else:
            trend = 'stable'

        dataset.append({
            'domain': 'gpu_timeseries',
            'timesteps': 3,
            't1': t1, 't2': t2, 't3': t3,
            'delta': delta_total,
            'trend': trend,
            'threshold_value': threshold,
            'action_text': action,
            'inputs': [[{'t': 1, 'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': t1}],
                       [{'t': 2, 'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': t2}],
                       [{'t': 3, 'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': t3}]],
            'thresholds': [{'sensor_type': 'temperature', 'comparison': '>=',
                            'threshold_value': threshold, 'action_text': action}],
            'output_essay': body,
            'timestamp': time.time(),
        })

    return dataset


def build_multidomain_dataset(n=50):
    """
    다중 도메인 시계열 essay 데이터셋.
    n: 다중 도메인 essay 수
    """
    dataset = []

    for i in range(n):
        # 3개 도메인의 랜덤 시계열
        gpu_t = [round(random.uniform(30.0, 70.0), 1) for _ in range(3)]
        battery_t = [round(random.uniform(10.5, 13.0), 1) for _ in range(3)]
        motor_t = [round(random.uniform(20.0, 80.0), 1) for _ in range(3)]

        gpu_threshold = round(random.uniform(40.0, 65.0), 1)
        battery_threshold = round(random.uniform(11.0, 12.5), 1)
        motor_threshold = round(random.uniform(40.0, 75.0), 1)

        gpu_history = [[gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=t, power_w=20.0)] for t in gpu_t]
        battery_history = [[gt.GpuSample(0, 'battery1', 'mock', temperature_c=t, power_w=0.0)] for t in battery_t]
        motor_history = [[gt.GpuSample(0, 'motor2', 'mock', temperature_c=t, power_w=0.0)] for t in motor_t]

        thresholds_dict = {
            'gpu': [gt.Threshold('temperature', '>=', gpu_threshold, 'Move quickly')],
            'battery': [gt.Threshold('temperature', '<=', battery_threshold, 'Return to base')],
            'motor': [gt.Threshold('temperature', '>=', motor_threshold, 'Stop motor')],
        }
        history_dict = {
            'gpu': gpu_history,
            'battery': battery_history,
            'motor': motor_history,
        }

        essay = gp.compose_essay_timeseries_multidomain(history_dict, thresholds_dict)

        # 추세 분석
        def trend(ts):
            d = ts[-1] - ts[0]
            return 'rising' if d > 1.0 else ('falling' if d < -1.0 else 'stable')

        dataset.append({
            'domain': 'multidomain',
            'timesteps': 3,
            'domains': ['gpu', 'battery', 'motor'],
            'trends': {'gpu': trend(gpu_t), 'battery': trend(battery_t), 'motor': trend(motor_t)},
            'inputs': history_dict,
            'thresholds': thresholds_dict,
            'output_essay': essay,
            'timestamp': time.time(),
        })

    return dataset


def main():
    print("Building large grounding dataset...")

    # 단일 시점 essay: 도메인당 100개 (총 300개)
    single = build_single_essay_dataset(n_per_domain=100)
    print(f"  Single essays: {len(single)} (battery + motor + gpu)")

    # 시계열 essay: 100개
    timeseries = build_timeseries_dataset(n=100)
    print(f"  Timeseries essays: {len(timeseries)}")

    # 다중 도메인 essay: 50개
    multidomain = build_multidomain_dataset(n=50)
    print(f"  Multidomain essays: {len(multidomain)}")

    all_essays = single + timeseries + multidomain

    output_path = '/home/kgh/grounding_dataset_large.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_essays, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_essays)} essays to {output_path}")


if __name__ == '__main__':
    main()
