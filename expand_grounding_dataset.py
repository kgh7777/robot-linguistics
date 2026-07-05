"""
expand_grounding_dataset.py

그라운딩 데이터셋 자동 확장. 9개 essay를 100-1000개로 확장.

확장 차원:
1. 값(value) 범위 - 각 임계치 주변으로 연속적인 값
2. 시계열 패턴 - rising, falling, stable, spike, dip
3. 다중 도메인 조합 - 다양한 도메인 집합
4. 비교 방향 - >=, <=
5. 액션/안정 케이스

출력: /home/kgh/grounding_dataset_extended.json
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


def build_extended_battery_dataset():
    """Battery essay 확장 - 값 범위 + 다양한 임계치."""
    dataset = []

    # 5개 임계치 시나리오
    threshold_scenarios = [
        (10.0, '<=', 'Return to base'),
        (11.0, '<=', 'Return to base'),
        (11.5, '<=', 'Return to base'),
        (12.0, '<=', 'Return to base'),
        (12.4, '<=', 'Return to base'),
    ]

    # 각 임계치마다 10개 값
    values = [9.0, 9.5, 10.0, 10.5, 11.0, 11.2, 11.5, 11.8, 12.0, 12.4]

    for threshold_value, comparison, action_text in threshold_scenarios:
        for value in values:
            readings = [gt.GpuSample(0, 'battery1', 'mock', temperature_c=value, power_w=0.0)]
            thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]
            essay = gp.compose_essay_universal(readings, thresholds)

            is_action = value <= threshold_value

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


def build_extended_motor_dataset():
    """Motor essay 확장 - 3개 임계치 시나리오 × 10개 값."""
    dataset = []

    threshold_scenarios = [
        (60.0, '>=', 'Reduce load'),
        (70.0, '>=', 'Reduce load'),
        (80.0, '>=', 'Stop motor'),
        (90.0, '>=', 'Emergency stop'),
    ]

    values = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]

    for threshold_value, comparison, action_text in threshold_scenarios:
        for value in values:
            readings = [gt.GpuSample(0, 'motor2', 'mock', temperature_c=value, power_w=0.0)]
            thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]
            essay = gp.compose_essay_universal(readings, thresholds)

            is_action = value >= threshold_value

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


def build_extended_gpu_dataset():
    """GPU essay 확장 - 4-way Radeon 클러스터, 다양한 임계치."""
    dataset = []

    # 3개 시점, 다양한 시나리오
    scenarios = [
        # (timestamps, threshold_temp, threshold_power)
        ([44.0, 46.0, 48.0], 40.0, 250.0),  # rising
        ([48.0, 46.0, 44.0], 40.0, 250.0),  # falling
        ([45.0, 45.0, 45.0], 40.0, 250.0),  # stable
        ([50.0, 55.0, 60.0], 40.0, 250.0),  # spike
        ([40.0, 38.0, 35.0], 40.0, 250.0),  # dip
    ]

    for temps, threshold_temp, threshold_power in scenarios:
        # 마지막 시점으로 단락 생성
        latest_temp = temps[-1]

        # GPU 단일 시점
        readings = [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=latest_temp, power_w=20.0)]
        thresholds = [gt.Threshold('temperature', '>=', threshold_temp, 'Move quickly')]
        essay = gp.compose_essay_universal(readings, thresholds)

        dataset.append({
            'domain': 'gpu',
            'timesteps': len(temps),
            'time_series': temps,
            'value': latest_temp,
            'comparison': '>=',
            'threshold_value': threshold_temp,
            'is_action': latest_temp >= threshold_temp,
            'action_text': 'Move quickly',
            'inputs': [{'sensor_id': 'card0', 'sensor_type': 'temperature', 'value': latest_temp, 'unit': 'C'}],
            'thresholds': [{'sensor_type': 'temperature', 'comparison': '>=', 'threshold_value': threshold_temp, 'action_text': 'Move quickly'}],
            'output_essay': essay,
            'timestamp': time.time(),
        })

    return dataset


def build_extended_multidomain_dataset():
    """다중 도메인 essay 확장 - 5가지 시나리오."""
    dataset = []

    scenarios = [
        # (gpu_temps, battery_temps, motor_temps, gpu_threshold, battery_threshold, motor_threshold)
        ([44.0, 46.0, 48.0], [12.5, 12.3, 12.4], [30.0, 32.0, 35.0], 40.0, 11.0, 80.0),  # GPU rising
        ([48.0, 46.0, 44.0], [11.5, 11.0, 10.5], [30.0, 32.0, 35.0], 40.0, 11.0, 80.0),  # battery falling
        ([50.0, 55.0, 60.0], [12.0, 11.5, 11.0], [60.0, 70.0, 85.0], 40.0, 11.0, 80.0),  # multiple alerts
        ([35.0, 35.0, 35.0], [12.5, 12.5, 12.5], [30.0, 30.0, 30.0], 40.0, 11.0, 80.0),  # all stable
        ([40.0, 42.0, 44.0], [11.5, 11.3, 11.1], [50.0, 60.0, 75.0], 40.0, 11.0, 80.0),  # borderline
    ]

    for gpu_temps, battery_temps, motor_temps, gpu_thr, bat_thr, mot_thr in scenarios:
        gpu_history = [[gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=t, power_w=20.0)] for t in gpu_temps]
        battery_history = [[gt.GpuSample(0, 'battery1', 'mock', temperature_c=t, power_w=0.0)] for t in battery_temps]
        motor_history = [[gt.GpuSample(0, 'motor2', 'mock', temperature_c=t, power_w=0.0)] for t in motor_temps]

        history = {'gpu': gpu_history, 'battery': battery_history, 'motor': motor_history}
        thresholds = {
            'gpu': [gt.Threshold('temperature', '>=', gpu_thr, 'Move quickly')],
            'battery': [gt.Threshold('temperature', '<=', bat_thr, 'Return to base')],
            'motor': [gt.Threshold('temperature', '>=', mot_thr, 'Stop motor')],
        }

        essay = gp.compose_essay_timeseries_multidomain(history, thresholds)

        dataset.append({
            'domain': 'multidomain',
            'timesteps': len(gpu_temps),
            'time_series': {
                'gpu': gpu_temps,
                'battery': battery_temps,
                'motor': motor_temps,
            },
            'inputs': {
                'gpu': [[{'value': s.temperature_c, 'unit': 'C'} for s in t] for t in gpu_history],
                'battery': [[{'value': s.temperature_c, 'unit': 'C'} for s in t] for t in battery_history],
                'motor': [[{'value': s.temperature_c, 'unit': 'C'} for s in t] for t in motor_history],
            },
            'thresholds': {
                'gpu': [{'comparison': '>=', 'threshold_value': gpu_thr, 'action_text': 'Move quickly'}],
                'battery': [{'comparison': '<=', 'threshold_value': bat_thr, 'action_text': 'Return to base'}],
                'motor': [{'comparison': '>=', 'threshold_value': mot_thr, 'action_text': 'Stop motor'}],
            },
            'output_essay': essay,
            'timestamp': time.time(),
        })

    return dataset


def main():
    print("Building extended grounding dataset...")

    battery = build_extended_battery_dataset()
    print(f"  Battery: {len(battery)} essays")

    motor = build_extended_motor_dataset()
    print(f"  Motor: {len(motor)} essays")

    gpu = build_extended_gpu_dataset()
    print(f"  GPU: {len(gpu)} essays")

    multidomain = build_extended_multidomain_dataset()
    print(f"  Multidomain: {len(multidomain)} essays")

    all_essays = battery + motor + gpu + multidomain

    output_path = '/home/kgh/grounding_dataset_extended.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_essays, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_essays)} essays to {output_path}")
    print(f"Breakdown: Battery={len(battery)}, Motor={len(motor)}, GPU={len(gpu)}, Multidomain={len(multidomain)}")


if __name__ == '__main__':
    main()
