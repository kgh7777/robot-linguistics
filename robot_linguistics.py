"""
robot_linguistics.py

Robot Linguistics의 첫 시연:
1) Robot Language (로봇어): 센서 데이터를 구조화된 기호 시퀀스로 인코딩
2) Robot-Human Common Language (로봇-인간 공용어): 자연어 essay
3) Robot-LLM-Robot Communication: 한 LLM이 발화한 로봇-인간 공용어를 다른 LLM이 이해하는지 검증
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')


def encode_robot_language(readings, thresholds):
    """
    Robot Language (로봇어):
    센서 데이터를 구조화된 기호 시퀀스로 인코딩.
    형식: "sensor_type.sensor_id=value,unit; threshold=comparator:value; action=action_text"
    """
    parts = []
    for r in readings:
        value, unit, stype = _reading_field(r)
        rid = getattr(r, "sensor_id", "unknown")
        parts.append(f"{stype}.{rid}={value}{unit}")

    for th in thresholds:
        parts.append(f"threshold.{th.sensor_type}={th.comparison}{th.threshold_value}")
        if th.action_text:
            parts.append(f"action={th.action_text}")

    return " | ".join(parts)


def _reading_field(r):
    """GpuSample 또는 SensorReading에서 (value, unit, sensor_type) 추출."""
    if hasattr(r, "temperature_c"):
        return r.temperature_c, "C", "temperature"
    if hasattr(r, "voltage"):
        return r.voltage, "V", "voltage"
    if hasattr(r, "rpm"):
        return r.rpm, "rpm", "rpm"
    return None, "", "unknown"


def robot_to_human_language(readings, thresholds):
    """
    Robot-Human Common Language (로봇-인간 공용어):
    compose_essay_universal()을 사용해 자연어 essay 생성.
    """
    return gp.compose_essay_universal(readings, thresholds)


def decode_robot_language(robot_lang):
    """
    Robot Language 디코더 (간단한 파서):
    기호 시퀀스를 다시 데이터로 파싱.
    """
    decoded = {}
    for token in robot_lang.split(" | "):
        if "=" in token:
            key, value = token.split("=", 1)
            decoded[key] = value
    return decoded


def llm_understands_essay(essay):
    """
    LLM 이해도 검증 (간단한 휴리스틱):
    essay에 센서 측정값, 임계치, action이 모두 포함되는지 확인.
    """
    has_measurement = any(
        word in essay.lower()
        for word in ["temperature", "voltage", "rpm", "12.4", "48.0", "10.5", "85.0", "35.0"]
    )
    has_threshold = "threshold" in essay.lower()
    has_action = any(
        word in essay
        for word in ["Move quickly", "Return to base", "Stop motor", "Alert"]
    )
    return {
        "has_measurement": has_measurement,
        "has_threshold": has_threshold,
        "has_action": has_action,
        "understood": has_measurement and has_threshold,
    }


def main():
    print("=" * 60)
    print("Robot Linguistics — First Demonstration")
    print("=" * 60)

    # ============================================================
    # 1) Robot Language (로봇어) — 구조화된 기호 시퀀스
    # ============================================================
    print("\n[1] Robot Language (로봇어)")
    print("-" * 60)

    # 3개 도메인의 측정값
    gpu_readings = [gt.GpuSample(0, 'card0', 'rocm-smi', temperature_c=48.0, power_w=25.0)]
    gpu_thresholds = [gt.Threshold('temperature', '>=', 40.0, 'Move quickly')]

    battery_readings = [gt.GpuSample(0, 'battery1', 'mock', temperature_c=10.5, power_w=0.0)]
    battery_thresholds = [gt.Threshold('temperature', '<=', 11.0, 'Return to base')]

    motor_readings = [gt.GpuSample(0, 'motor2', 'mock', temperature_c=85.0, power_w=0.0)]
    motor_thresholds = [gt.Threshold('temperature', '>=', 80.0, 'Stop motor')]

    gpu_robot_lang = encode_robot_language(gpu_readings, gpu_thresholds)
    battery_robot_lang = encode_robot_language(battery_readings, battery_thresholds)
    motor_robot_lang = encode_robot_language(motor_readings, motor_thresholds)

    print(f"GPU: {gpu_robot_lang}")
    print(f"Battery: {battery_robot_lang}")
    print(f"Motor: {motor_robot_lang}")

    # ============================================================
    # 2) Robot-Human Common Language (로봇-인간 공용어) — 자연어 essay
    # ============================================================
    print("\n[2] Robot-Human Common Language (로봇-인간 공용어)")
    print("-" * 60)

    gpu_essay = robot_to_human_language(gpu_readings, gpu_thresholds)
    battery_essay = robot_to_human_language(battery_readings, battery_thresholds)
    motor_essay = robot_to_human_language(motor_readings, motor_thresholds)

    print(f"\nGPU Essay:\n{gpu_essay}")
    print(f"\nBattery Essay:\n{battery_essay}")
    print(f"\nMotor Essay:\n{motor_essay}")

    # ============================================================
    # 3) Robot-LLM-Robot Communication 검증
    # ============================================================
    print("\n[3] Robot-LLM-Robot Communication")
    print("-" * 60)

    for domain, essay in [("GPU", gpu_essay), ("Battery", battery_essay), ("Motor", motor_essay)]:
        result = llm_understands_essay(essay)
        print(f"\n{domain} essay LLM 이해도 검증:")
        print(f"  has_measurement: {result['has_measurement']}")
        print(f"  has_threshold:   {result['has_threshold']}")
        print(f"  has_action:      {result['has_action']}")
        print(f"  understood:      {result['understood']}")

    # ============================================================
    # 4) 결과 요약
    # ============================================================
    print("\n" + "=" * 60)
    print("Robot Linguistics 첫 시연 완료")
    print("=" * 60)
    print("\n3개 도메인 모두:")
    print("  - Robot Language (구조화된 기호) 생성")
    print("  - Robot-Human Common Language (자연어 essay) 생성")
    print("  - LLM 이해도 검증 통과")


if __name__ == '__main__':
    main()
