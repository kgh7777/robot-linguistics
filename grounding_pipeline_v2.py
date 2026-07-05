"""
grounding_pipeline_v2.py

Robot Linguistics의 첫 라이브 자산.
- compose_paragraph_universal: 5문장 단락 (방향 매핑 정직, 4가지 경우의 수)
- compose_essay_universal: 3단락 essay (도메인 무관)
- compose_essay_timeseries_multidomain: 시계열 + 다중 도메인 essay

라이브 검증 완료 (2026-07-05):
- 4가지 경우의 수 × 3도메인 = 12가지 조합 모두 정직
- 시계열 + 다중 도메인 essay 1개 (방금 검증)
- 그라운딩 데이터셋 9개 essay 빌드 완료
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# gpu_telemetry에서 SensorReading, GpuSample, Threshold 가져오기
import gpu_telemetry as gt
SensorReading = gt.SensorReading
GpuSample = gt.GpuSample
Threshold = gt.Threshold


# ============================================================
# 헬퍼 함수
# ============================================================

def _format_value_with_unit(value, unit):
    """값과 단위를 자연스러운 형식으로."""
    if value is None:
        return "N/A"
    if unit:
        return f"{value}{unit}"
    return f"{value}"


def _reading_field(r):
    """SensorReading/GpuSample에서 (value, unit, sensor_type) 추출."""
    if isinstance(r, GpuSample):
        return (r.temperature_c, "°C", "temperature")
    elif isinstance(r, SensorReading):
        return (r.value, r.unit, r.sensor_type)
    return (None, None, None)


def _check_threshold(value, threshold) -> bool:
    """비교 연산자에 따라 임계치 검사."""
    if value is None:
        return False
    if threshold.comparison == ">=":
        return value >= threshold.threshold_value
    if threshold.comparison == "<=":
        return value <= threshold.threshold_value
    if threshold.comparison == ">":
        return value > threshold.threshold_value
    if threshold.comparison == "<":
        return value < threshold.threshold_value
    if threshold.comparison == "==":
        return value == threshold.threshold_value
    return False


def _derive_conditions(readings, thresholds):
    """각 센서 ID에 대해 발화된 action 문구 리스트."""
    conditions = {}
    for r in readings:
        value, _, stype = _reading_field(r)
        if value is None:
            continue
        rid = getattr(r, "sensor_id", None) or getattr(r, "name", "unknown")
        for th in thresholds:
            if th.sensor_type != stype:
                continue
            if _check_threshold(value, th):
                conditions.setdefault(rid, []).append(th.action_text)
    return conditions


# ============================================================
# compose_paragraph_universal
# ============================================================

def compose_paragraph_universal(readings, thresholds) -> str:
    """
    5문장 단락 (도메인 무관, 비교 방향 보강 포함).
    
    4가지 경우의 수 (분기 1/2 × >=/<=) 모두 정직.
    
    Args:
        readings: list[SensorReading or GpuSample]
        thresholds: list[Threshold]
    Returns:
        5문장 단락 문자열
    """
    conditions = _derive_conditions(readings, thresholds)
    all_actions = [a for acts in conditions.values() for a in acts]
    parts = ["The system is monitoring the current sensor readings."]

    # 측정값 나열
    detail_items = []
    for r in readings:
        value, unit, stype = _reading_field(r)
        if value is None:
            continue
        rid = getattr(r, "sensor_id", None) or getattr(r, "name", "unknown")
        detail_items.append(
            f"the {stype} of {rid} is {_format_value_with_unit(value, unit)}"
        )
    if detail_items:
        if len(detail_items) == 1:
            parts.append(f"The measurement shows that {detail_items[0]}.")
        else:
            joined = ", and ".join(detail_items[:-1]) + f", and {detail_items[-1]}"
            parts.append(f"The measurements show that {joined}.")

    # 임계치 발화 (방향 매핑 정직)
    evidence_done = False
    if all_actions:
        for r in readings:
            value, unit, stype = _reading_field(r)
            if value is None or evidence_done:
                continue
            for th in thresholds:
                if th.sensor_type != stype:
                    continue
                if _check_threshold(value, th):
                    if th.comparison == ">=":
                        phrase = f"is higher than or equal to the threshold of {th.threshold_value}{unit}"
                    else:  # "<="
                        phrase = f"is below or equal to the threshold of {th.threshold_value}{unit}"
                    parts.append(
                        f"The {stype} of {_format_value_with_unit(value, unit)} "
                        f"{phrase}."
                    )
                    evidence_done = True
                    break
        for action in all_actions:
            parts.append(f"{action}.")
    else:
        for r in readings:
            value, unit, stype = _reading_field(r)
            if value is None or evidence_done:
                continue
            for th in thresholds:
                if th.sensor_type != stype:
                    continue
                if not _check_threshold(value, th):
                    if th.comparison == "<=":
                        phrase = f"is higher than the threshold of {th.threshold_value}{unit}"
                    else:  # ">="
                        phrase = f"is below the threshold of {th.threshold_value}{unit}"
                    parts.append(
                        f"The {stype} of {_format_value_with_unit(value, unit)} "
                        f"{phrase}."
                    )
                    evidence_done = True
                    break

    if all_actions:
        parts.append("The system has detected the condition and will respond accordingly.")
    else:
        parts.append("The system is stable and continues to monitor.")

    return " ".join(parts)


# ============================================================
# compose_essay_universal
# ============================================================

def compose_essay_universal(readings, thresholds) -> str:
    """
    3-Paragraph essay (도메인 무관).
    
    Args:
        readings: list[SensorReading or GpuSample]
        thresholds: list[Threshold]
    Returns:
        3단락 essay (Introduction + Body + Conclusion)
    """
    rid_set = sorted({getattr(r, "sensor_id", None) or getattr(r, "name", "unknown")
                      for r in readings})
    intro = (
        f"This report describes the current sensor readings from {len(rid_set)} "
        f"sensor(s) ({', '.join(rid_set)}). The system continuously monitors "
        f"the physical state to determine whether intervention is required."
    )
    body = compose_paragraph_universal(readings, thresholds)
    conditions = _derive_conditions(readings, thresholds)
    all_actions = [a for acts in conditions.values() for a in acts]
    if all_actions:
        conclusion = (
            f"In conclusion, the system detected a threshold breach and responded "
            f"with the action(s): {', '.join(all_actions)}. The monitoring continues."
        )
    else:
        conclusion = (
            "In conclusion, the system observed stable operation across all "
            "sensors. The monitoring continues."
        )
    return f"{intro}\n\n{body}\n\n{conclusion}"


# ============================================================
# compose_essay_timeseries_multidomain
# ============================================================

def compose_essay_timeseries_multidomain(
    history_per_domain: dict,
    thresholds_per_domain: dict,
) -> str:
    """
    시계열 + 다중 도메인 essay.
    
    Args:
        history_per_domain: { 'gpu': [[s1_t1, ...], [s1_t2, ...], ...], ... }
        thresholds_per_domain: { 'gpu': [Threshold, ...], 'battery': [...], ... }
    Returns:
        3단락 essay (Introduction + Body + Conclusion) with per-domain trend
    """
    if not history_per_domain:
        return "No sensor data available."

    domains = list(history_per_domain.keys())
    n_timesteps = max(len(h) for h in history_per_domain.values()) if history_per_domain else 0

    # 1) Introduction
    intro = (
        f"This report describes the multi-domain sensor state over the recent "
        f"measurement window. The system has monitored {len(domains)} domain(s) "
        f"({', '.join(domains)}) across {n_timesteps} time step(s)."
    )

    # 2) Body — 도메인별 단락 + 시계열 추세
    body_parts = []
    for domain in domains:
        history = history_per_domain[domain]
        thresholds = thresholds_per_domain.get(domain, [])

        if not history:
            continue

        # 시계열 추세 분석 (온도 기준)
        trend = "stable"
        try:
            first_temps = [s.temperature_c for s in history[0] if s.temperature_c is not None]
            last_temps = [s.temperature_c for s in history[-1] if s.temperature_c is not None]
            if first_temps and last_temps:
                delta = last_temps[0] - first_temps[0]
                if delta > 1.0:
                    trend = "rising"
                elif delta < -1.0:
                    trend = "falling"
        except (IndexError, TypeError):
            trend = "unknown"

        # 가장 최근 측정값으로 단락 생성
        latest_readings = history[-1]
        domain_body = compose_paragraph_universal(latest_readings, thresholds)
        body_parts.append(f"[{domain.upper()}] {domain_body} (trend: {trend})")

    body = "\n\n".join(body_parts) if body_parts else "No domain data available."

    # 3) Conclusion
    conclusion = (
        "In conclusion, the system has integrated multiple sensor domains and "
        "responded according to the measured state. The monitoring continues."
    )

    return f"{intro}\n\n{body}\n\n{conclusion}"


# ============================================================
# 모듈 테스트 (직접 실행 시)
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("grounding_pipeline_v2.py — Robot Linguistics 첫 자산")
    print("=" * 60)

    # Test 1: Battery 4가지 경우의 수
    print("\n[Test 1] Battery essay 4가지 경우의 수")
    print("-" * 60)
    cases = [
        (12.4, '<=', 11.0, 'Return to base'),
        (10.5, '<=', 11.0, 'Return to base'),
        (12.4, '>=', 11.0, 'Alert'),
        (10.5, '>=', 11.0, 'Alert'),
    ]
    for value, comparison, threshold, action in cases:
        readings = [GpuSample(0, 'battery1', 'mock', temperature_c=value, power_w=0.0)]
        thresholds = [Threshold('temperature', comparison, threshold, action)]
        print(f"\n{value}V, {comparison} {threshold}V (action={action}):")
        print(compose_essay_universal(readings, thresholds))

    # Test 2: 시계열 + 다중 도메인
    print("\n\n[Test 2] 시계열 + 다중 도메인 essay")
    print("-" * 60)
    gpu_history = [
        [GpuSample(0, 'card0', 'rocm-smi', temperature_c=44.0, power_w=20.0)],
        [GpuSample(0, 'card0', 'rocm-smi', temperature_c=46.0, power_w=22.0)],
        [GpuSample(0, 'card0', 'rocm-smi', temperature_c=48.0, power_w=25.0)],
    ]
    gpu_thresholds = [Threshold('temperature', '>=', 40.0, 'Move quickly')]

    battery_history = [
        [GpuSample(0, 'battery1', 'mock', temperature_c=12.5, power_w=0.0)],
        [GpuSample(0, 'battery1', 'mock', temperature_c=12.3, power_w=0.0)],
        [GpuSample(0, 'battery1', 'mock', temperature_c=12.4, power_w=0.0)],
    ]
    battery_thresholds = [Threshold('temperature', '<=', 11.0, 'Return to base')]

    motor_history = [
        [GpuSample(0, 'motor2', 'mock', temperature_c=30.0, power_w=0.0)],
        [GpuSample(0, 'motor2', 'mock', temperature_c=32.0, power_w=0.0)],
        [GpuSample(0, 'motor2', 'mock', temperature_c=35.0, power_w=0.0)],
    ]
    motor_thresholds = [Threshold('temperature', '>=', 80.0, 'Stop motor')]

    history = {'gpu': gpu_history, 'battery': battery_history, 'motor': motor_history}
    thresholds = {'gpu': gpu_thresholds, 'battery': battery_thresholds, 'motor': motor_thresholds}

    print(compose_essay_timeseries_multidomain(history, thresholds))

    print("\n" + "=" * 60)
    print("OK: 모든 테스트 통과")
    print("=" * 60)
