"""
build_grounding_dataset_v2.py

그라운딩 데이터셋을 500-1000개 essay로 확장.
다양한 도메인, 임계치, 시나리오 포함.
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
        'sensor_ids': ['card0', 'card1', 'card2', 'card3'],
        'value_range': (30.0, 90.0),  # 30-90°C
        'unit': 'C',
        'action_texts': ['Move quickly', 'Reduce load', 'Free memory', 'Check fan'],
    },
    'battery': {
        'sensor_ids': ['battery1', 'battery2'],
        'value_range': (9.0, 13.0),  # 9-13V
        'unit': 'V',
        'action_texts': ['Return to base', 'Recharge', 'Reduce load', 'Check wiring'],
    },
    'motor': {
        'sensor_ids': ['motor0', 'motor1', 'motor2'],
        'value_range': (15.0, 90.0),  # 15-90°C
        'unit': 'C',
        'action_texts': ['Stop motor', 'Reduce load', 'Cool down', 'Lubricate'],
    },
    'lidar': {
        'sensor_ids': ['lidar_front', 'lidar_back'],
        'value_range': (0.1, 10.0),  # 0.1-10m
        'unit': 'm',
        'action_texts': ['Stop immediately', 'Turn around', 'Slow down', 'Reroute'],
    },
}


def generate_random_essay():
    """랜덤 essay 1개 생성."""
    domain = random.choice(list(DOMAINS.keys()))
    config = DOMAINS[domain]
    sensor_id = random.choice(config['sensor_ids'])
    value = random.uniform(*config['value_range'])
    action_text = random.choice(config['action_texts'])

    # 임계치는 value의 ±20% 범위에서 랜덤
    threshold_value = value * random.uniform(0.7, 1.3)
    comparison = random.choice(['>=', '<='])
    is_action = random.choice([True, False])

    # action 비활성화 시 action_text는 빈 문자열
    if not is_action:
        action_text = ''

    # Reading + Threshold 생성
    readings = [gt.GpuSample(0, sensor_id, 'mock', temperature_c=value, power_w=0.0)]
    thresholds = [gt.Threshold('temperature', comparison, threshold_value, action_text)]

    # Essay 생성
    try:
        essay = gp.compose_essay_universal(readings, thresholds)
    except Exception as e:
        return None

    return {
        'domain': domain,
        'timesteps': 1,
        'sensor_id': sensor_id,
        'value': value,
        'comparison': comparison,
        'threshold_value': threshold_value,
        'is_action': is_action,
        'action_text': action_text,
        'inputs': [{'sensor_id': sensor_id, 'sensor_type': 'temperature', 'value': value, 'unit': config['unit']}],
        'thresholds': [{'sensor_type': 'temperature', 'comparison': comparison, 'threshold_value': threshold_value, 'action_text': action_text}],
        'output_essay': essay,
        'timestamp': time.time(),
    }


def main():
    n_essays = 500
    print(f"Building expanded grounding dataset ({n_essays} essays)...")

    dataset = []
    failures = 0
    while len(dataset) < n_essays:
        result = generate_random_essay()
        if result is not None:
            dataset.append(result)
        else:
            failures += 1
        if failures > n_essays * 2:  # 너무 많은 실패
            break
        if len(dataset) % 50 == 0 and len(dataset) > 0:
            print(f"  Generated {len(dataset)} essays...")

    # 메타데이터 추가
    output = {
        'metadata': {
            'created_at': time.time(),
            'total_essays': len(dataset),
            'domains': list(DOMAINS.keys()),
            'generator': 'build_grounding_dataset_v2.py',
            'purpose': 'Robot Linguistics research - first large-scale grounding dataset',
        },
        'essays': dataset,
    }

    output_path = '/home/kgh/grounding_dataset_v2.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(dataset)} essays to {output_path}")

    # 도메인별 통계
    print("\n=== Domain distribution ===")
    domain_counts = {}
    for e in dataset:
        domain_counts[e['domain']] = domain_counts.get(e['domain'], 0) + 1
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c} essays ({c/len(dataset)*100:.1f}%)")

    # action 발화 비율
    action_count = sum(1 for e in dataset if e['is_action'])
    print(f"\n=== Action distribution ===")
    print(f"  With action: {action_count} ({action_count/len(dataset)*100:.1f}%)")
    print(f"  Without action: {len(dataset) - action_count} ({(len(dataset)-action_count)/len(dataset)*100:.1f}%)")


if __name__ == '__main__':
    main()
