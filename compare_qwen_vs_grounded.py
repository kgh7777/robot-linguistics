"""
compare_qwen_vs_grounded.py

Qwen mode vs Sensor-grounded mode 비교 평가.
같은 입력 (센서 데이터) 으로 두 모드의 발화를 생성하고 비교.

평가 항목:
1. 수치 인용 정확도 (measurement value 정확히 등장?)
2. 방향 정직성 (higher/lower 방향이 맞나?)
3. 환각 여부 (입력에 없는 수치 등장?)
4. 결정성 (같은 입력에 같은 출력?)
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import importlib
import requests

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


def query_qwen2_5(sensor_description: str, host: str = "192.168.1.171", port: int = 11434) -> str:
    """
    Qwen2.5에게 센서 상황 설명 후 발화 요청.
    """
    prompt = f"""Given the following sensor reading, write a single English sentence describing the situation:
{sensor_description}

Write only the English sentence, no other text."""

    try:
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json={
                "model": "qwen2.5:32b-q8_0",
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        return f"[Qwen error: {e}]"
    return ""


def generate_grounded_essay(sensor_id: str, value: float, comparison: str, threshold: float, action: str) -> str:
    """지금 만든 센서 그라운딩 essay 합성."""
    readings = [gt.GpuSample(0, sensor_id, 'mock', temperature_c=value, power_w=0.0)]
    thresholds = [gt.Threshold('temperature', comparison, threshold, action)]
    return gp.compose_essay_universal(readings, thresholds)


def check_measurement_accuracy(essay: str, expected_value: float) -> bool:
    """수치 인용 정확도 검사."""
    return str(expected_value) in essay or f"{expected_value:.1f}" in essay


def check_direction(essay: str, value: float, threshold: float, comparison: str) -> str:
    """
    방향 정직성 검사.
    Returns: 'correct' | 'incorrect' | 'ambiguous'
    """
    value_above = value > threshold
    value_below = value < threshold

    # higher than / above the threshold
    higher_in_essay = ('higher than' in essay.lower() or 'above' in essay.lower())
    # below the threshold
    below_in_essay = ('below' in essay.lower() or 'lower than' in essay.lower())

    if comparison == '<=':
        if value_above:
            return 'correct' if higher_in_essay else 'incorrect'
        else:  # value_below
            return 'correct' if below_in_essay else 'incorrect'
    elif comparison == '>=':
        if value_above:
            return 'correct' if higher_in_essay else 'incorrect'
        else:  # value_below
            return 'correct' if below_in_essay else 'incorrect'
    return 'ambiguous'


def check_hallucination(essay: str, allowed_values: list) -> list:
    """환각 검사 — 입력에 없는 수치가 essay에 있는지."""
    import re
    # essay에서 숫자 추출
    numbers_in_essay = re.findall(r'\d+\.?\d*', essay)
    # 입력에 있는 숫자만 화이트리스트
    allowed = set(str(v) for v in allowed_values) | {f"{v:.1f}" for v in allowed_values}
    # 환각 = essay에 있지만 allowed에 없는 숫자
    hallucinations = [n for n in numbers_in_essay if n not in allowed]
    return hallucinations


def main():
    """
    4가지 경우의 수에 대해 두 모드 비교.
    """
    test_cases = [
        # (sensor_id, value, comparison, threshold, action, description)
        ('battery1', 12.4, '<=', 11.0, 'Return to base', 'battery voltage 12.4V, threshold 11.0V (low)'),
        ('battery1', 10.5, '<=', 11.0, 'Return to base', 'battery voltage 10.5V, threshold 11.0V (low)'),
        ('battery1', 12.4, '>=', 11.0, 'Alert', 'battery voltage 12.4V, threshold 11.0V (high)'),
        ('battery1', 10.5, '>=', 11.0, 'Alert', 'battery voltage 10.5V, threshold 11.0V (high)'),
    ]

    results = []

    for sensor_id, value, comparison, threshold, action, description in test_cases:
        print(f"\n=== {description} ===")
        print(f"Value: {value}, Comparison: {comparison}, Threshold: {threshold}, Action: {action}")

        # 1) Sensor-grounded essay 생성
        grounded_essay = generate_grounded_essay(sensor_id, value, comparison, threshold, action)
        print(f"\n[Sensor-grounded mode]")
        print(f"  Essay: {grounded_essay[:200]}...")

        # 2) Qwen essay 생성
        qwen_essay = query_qwen2_5(description)
        print(f"\n[Qwen2.5 mode]")
        print(f"  Essay: {qwen_essay[:200]}...")

        # 3) 평가
        # 수치 인용 정확도
        grounded_value_ok = check_measurement_accuracy(grounded_essay, value)
        qwen_value_ok = check_measurement_accuracy(qwen_essay, value)

        # 방향 정직성
        grounded_direction = check_direction(grounded_essay, value, threshold, comparison)
        qwen_direction = check_direction(qwen_essay, value, threshold, comparison)

        # 환각
        allowed = [value, threshold, 11.0, 10.5, 12.4]
        grounded_hallucinations = check_hallucination(grounded_essay, allowed)
        qwen_hallucinations = check_hallucination(qwen_essay, allowed)

        result = {
            'description': description,
            'value': value,
            'comparison': comparison,
            'threshold': threshold,
            'action': action,
            'grounded': {
                'essay': grounded_essay,
                'value_accuracy': grounded_value_ok,
                'direction': grounded_direction,
                'hallucinations': grounded_hallucinations,
            },
            'qwen': {
                'essay': qwen_essay,
                'value_accuracy': qwen_value_ok,
                'direction': qwen_direction,
                'hallucinations': qwen_hallucinations,
            },
        }
        results.append(result)

        print(f"\n[Evaluation]")
        print(f"  Sensor-grounded: value_ok={grounded_value_ok}, direction={grounded_direction}, hallucinations={grounded_hallucinations}")
        print(f"  Qwen2.5:         value_ok={qwen_value_ok}, direction={qwen_direction}, hallucinations={qwen_hallucinations}")

    # 결과 요약
    print("\n\n=== Summary ===")
    grounded_correct = sum(1 for r in results if r['grounded']['value_accuracy'] and r['grounded']['direction'] == 'correct' and not r['grounded']['hallucinations'])
    qwen_correct = sum(1 for r in results if r['qwen']['value_accuracy'] and r['qwen']['direction'] == 'correct' and not r['qwen']['hallucinations'])
    print(f"Sensor-grounded: {grounded_correct}/{len(results)} cases all correct")
    print(f"Qwen2.5:         {qwen_correct}/{len(results)} cases all correct")

    # JSON 저장
    output_path = '/home/kgh/qwen_vs_grounded.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
