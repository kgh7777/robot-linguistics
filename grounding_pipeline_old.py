"""
grounding_pipeline.py

gpu_telemetry.py + 정직한 프롬프트 → Qwen2.5 호출 → 한국어/영어 한 문장씩 발화

목표: 모델이 *지금 측정한 GPU 물리 수치를 그대로 인용*하는 한국어/영어 문장을 생성하는지 검증.
- 시적/형이상학적 프레이밍 ❌
- "기계 고유 사유" 같은 수식어 ❌
- "입력에 없는 수치를 쓰지 마라"는 가드 포함
- 출력은 한국어 한 문장, 영어 한 문장 (둘 다 같은 물리 사실을 인용)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field

import requests

import gpu_telemetry as gt

log = logging.getLogger("grounding_pipeline")

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:2242/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "qwen2.5-32b-instruct-q8_0")
DEFAULT_TIMEOUT = (5, 60)  # (connect, read) — read는 60초까지


# ---------------------------------------------------------------------------
# 시스템 프롬프트 — 정직한 텔레메트리 보고서
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "너는 GPU 텔레메트리 보고서 작성기다.\n"
    "역할:\n"
    "  - 너에게 주어지는 텍스트는 방금 측정한 4개 Radeon GPU의 물리 수치와, "
    "그 수치로부터 외부 함수가 파생한 조건 플래그(action 문구)다.\n"
    "  - 너는 한국어 한 문장과 영어 한 문장을 만들어 출력한다.\n"
    "  - 두 문장 모두 같은 물리 사실을 인용하고, 같은 조건 플래그가 True면 "
    "그 action 문구를 포함해야 한다.\n"
    "엄격한 규칙:\n"
    "  (1) 입력에 있는 수치(온도, 전력, VRAM, 사용률)와 단위(C, W, MiB, %)를 그대로 인용하라.\n"
    "  (2) 입력에 없는 수치, 단위, 디바이스 번호를 만들지 마라.\n"
    "  (2a) 절대 학습 데이터의 '전형적인' 수치(예: idle 1.0%, 15.0% 같은 일반적인 사용률)를 "
    "추측하여 인용하지 마라. 입력에 있는 값만 인용하라.\n"
    "  (3) 은유, 형이상학, 시적 표현을 쓰지 마라.\n"
    "  (4) '기계의 사유', '자아', '초월' 같은 메타 담론을 쓰지 마라.\n"
    "  (5) 출력 외의 다른 텍스트(서론, 해설, 주석)를 쓰지 마라.\n"
    "  (6) action 문구가 주어지면, 그 문구를 KO/EN 양쪽 문장에 그대로 포함하라. "
    "다시 쓰지 말고 원문 그대로 사용하라.\n"
    "출력 형식(정확히 이 형태):\n"
    "  KO: <한국어 한 문장>\n"
    "  EN: <영어 한 문장>\n"
)


USER_PROMPT_TEMPLATE = (
    "방금 측정한 4-way GPU 텔레메트리 벡터는 다음과 같다.\n"
    "각 카드의 필드는 T=온도(°C), P=전력(W), V=VRAM 사용량(MiB), U=GPU 사용률(%)이다.\n"
    "\n"
    "{vector}\n"
    "\n"
    "위 측정값에 대해 외부 함수가 파생한 조건 플래그와 action 문구는 다음과 같다.\n"
    "{conditions}\n"
    "\n"
    "위 측정값과 파생 조건/action을 근거로 한국어 한 문장과 영어 한 문장을 작성하라. "
    "수치와 단위는 입력값과 정확히 일치해야 하고, "
    "action 문구는 주어지면 원문 그대로 포함하라."
)


# ---------------------------------------------------------------------------
# 외부 함수: 하드웨어 수치 → 파생 조건 플래그 + action 문구
# 사용자가 말한 "하드웨어 수치 - 함수 - token - 영어 문장" 공식의 '함수' 부분.
# ---------------------------------------------------------------------------

# 기본 임계치 (CLI에서 덮어쓸 수 있음)
DEFAULT_THRESHOLDS = {
    "temp_high_c": 70.0,   # 온도가 이 값 이상이면 "move quickly"
    "power_high_w": 250.0, # 전력이 이 값 이상이면 "reduce load"
    "vram_high_mib": 14000.0,  # VRAM이 이 값 이상이면 "free memory"
}


def derive_conditions(samples, thresholds=None):
    """
    측정값 리스트를 받아 파생 조건 플래그 + action 문구 생성.
    반환 형태:
        {
          "per_card": {
              0: {"temp_high": True, "action": "Move quickly", "rule": "T=72.0C >= 70.0C"},
              ...
          },
          "summary_lines": [
              "card0: T=72.0C >= 70.0C → action: 'Move quickly'",
              ...
          ],
          "any_action": True/False,   # 어떤 카드든 action이 있으면 True
          "all_actions": [...]        # 모든 action 문구 (KO/EN에 포함용)
        }
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    per_card = {}
    summary_lines = []
    all_actions = []
    any_action = False

    for s in samples:
        flags = {}
        rules = []

        # 온도 임계
        if s.temperature_c is not None and s.temperature_c >= thresholds["temp_high_c"]:
            flags["temp_high"] = True
            flags["action"] = "Move quickly"
            rules.append(
                f"T={s.temperature_c:.1f}C >= {thresholds['temp_high_c']:.1f}C"
            )
            all_actions.append("Move quickly")
            any_action = True

        # 전력 임계
        if s.power_w is not None and s.power_w >= thresholds["power_high_w"]:
            flags["power_high"] = True
            flags["action"] = "Reduce load"
            rules.append(
                f"P={s.power_w:.1f}W >= {thresholds['power_high_w']:.1f}W"
            )
            all_actions.append("Reduce load")
            any_action = True

        # VRAM 임계
        if s.vram_used_mib is not None and s.vram_used_mib >= thresholds["vram_high_mib"]:
            flags["vram_high"] = True
            flags["action"] = "Free memory"
            rules.append(
                f"V={s.vram_used_mib:.1f}MiB >= {thresholds['vram_high_mib']:.1f}MiB"
            )
            all_actions.append("Free memory")
            any_action = True

        per_card[s.index] = {"flags": flags, "rules": rules}
        if rules:
            summary_lines.append(
                f"card{s.index}: {'; '.join(rules)} → action(s): "
                f"{', '.join(set(flags.get(k) for k in flags if k.endswith('action') and flags[k]))}"
            )

    if not any_action:
        summary_lines.append("(어떤 임계치도 초과하지 않음 — action 없음)")

    return {
        "per_card": per_card,
        "summary_lines": summary_lines,
        "any_action": any_action,
        "all_actions": all_actions,
        "thresholds_used": dict(thresholds),
    }


def conditions_to_prompt_text(cond):
    """
    derive_conditions()의 결과를 LLM 입력용 텍스트로 직렬화.
    """
    th = cond["thresholds_used"]
    lines = [
        f"임계치: T>={th['temp_high_c']:.1f}C → 'Move quickly', "
        f"P>={th['power_high_w']:.1f}W → 'Reduce load', "
        f"V>={th['vram_high_mib']:.1f}MiB → 'Free memory'",
    ]
    lines.extend(cond["summary_lines"])
    if cond["all_actions"]:
        # 중복 제거 + 순서 유지
        seen = set()
        unique = []
        for a in cond["all_actions"]:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        lines.append(f"포함할 action 문구: {unique}")
    else:
        lines.append("포함할 action 문구: (없음 — 임계치 미만)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 결과 컨테이너
# ---------------------------------------------------------------------------

@dataclass
class GroundingAttempt:
    vector_input: str            # 입력 텔레메트리 벡터 문자열
    conditions_text: str         # LLM에 전달된 파생 조건 텍스트
    thresholds: dict             # 사용된 임계치
    raw_response: str            # 모델의 원본 응답 전체
    ko_sentence: str             # 추출된 한국어 문장
    en_sentence: str             # 추출된 영어 문장
    http_status: int

    def to_dict(self):
        return {
            "vector_input": self.vector_input,
            "conditions_text": self.conditions_text,
            "thresholds": self.thresholds,
            "raw_response": self.raw_response,
            "ko_sentence": self.ko_sentence,
            "en_sentence": self.en_sentence,
            "http_status": self.http_status,
        }


# ---------------------------------------------------------------------------
# 응답 파싱
# ---------------------------------------------------------------------------

_KO_RE = re.compile(r"^\s*KO\s*[:：]\s*(.+?)\s*$", re.M)
_EN_RE = re.compile(r"^\s*EN\s*[:：]\s*(.+?)\s*$", re.M)


def parse_ko_en(raw: str) -> tuple[str, str]:
    """
    모델 출력에서 "KO: ..." / "EN: ..." 한 줄씩을 추출.
    둘 중 하나라도 없으면 빈 문자열 반환.
    """
    ko_m = _KO_RE.search(raw)
    en_m = _EN_RE.search(raw)
    ko = ko_m.group(1).strip() if ko_m else ""
    en = en_m.group(1).strip() if en_m else ""
    return ko, en


# ---------------------------------------------------------------------------
# 핵심 호출
# ---------------------------------------------------------------------------

def call_qwen(
    vector_str: str,
    conditions_text: str = "",
    *,
    url: str = DEFAULT_LLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,   # 그라운딩 태스크라서 결정적
    timeout = DEFAULT_TIMEOUT,
) -> GroundingAttempt:
    """
    1) gpu_telemetry.to_vector_string()가 만들어준 벡터 문자열과
    2) derive_conditions()가 만든 파생 조건 텍스트를 받아
    3) Qwen2.5에 정직한 프롬프트를 보내
    4) KO/EN 한 문장씩 추출해 GroundingAttempt로 반환.
    """
    user_content = USER_PROMPT_TEMPLATE.format(
        vector=vector_str, conditions=conditions_text or "(파생 조건 없음)"
    )
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        log.error("Qwen 호출 실패: %s", e)
        return GroundingAttempt(
            vector_input=vector_str,
            conditions_text=conditions_text,
            thresholds={},
            raw_response=f"<request error: {e}>",
            ko_sentence="",
            en_sentence="",
            http_status=0,
        )

    if r.status_code != 200:
        return GroundingAttempt(
            vector_input=vector_str,
            conditions_text=conditions_text,
            thresholds={},
            raw_response=r.text,
            ko_sentence="",
            en_sentence="",
            http_status=r.status_code,
        )

    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as e:
        return GroundingAttempt(
            vector_input=vector_str,
            conditions_text=conditions_text,
            thresholds={},
            raw_response=f"<bad response shape: {e}>: {r.text[:200]}",
            ko_sentence="",
            en_sentence="",
            http_status=r.status_code,
        )

    ko, en = parse_ko_en(content)
    return GroundingAttempt(
        vector_input=vector_str,
        conditions_text=conditions_text,
        thresholds={},
        raw_response=content,
        ko_sentence=ko,
        en_sentence=en,
        http_status=r.status_code,
    )


# ---------------------------------------------------------------------------
# 범용 센서 프레임워크 (도메인 무관: GPU/배터리/모터/LiDAR/카메라/마이크)
# ---------------------------------------------------------------------------

import math
from collections import namedtuple

# 경량 데이터 구조: dataclass 대신 namedtuple 사용 (의존성 0)
# fields: sensor_id, sensor_type, value, unit, timestamp
SensorReading = namedtuple("SensorReading",
                            ["sensor_id", "sensor_type", "value", "unit", "timestamp"])
# 기본값 있는 변형
def make_reading(sensor_id, sensor_type, value, unit, timestamp=0.0):
    return SensorReading(sensor_id, sensor_type, value, unit, timestamp)

# 임계치 정의: (sensor_type, comparison, threshold_value, action_text)
#   comparison: ">=", "<=", ">", "<", "=="
Threshold = namedtuple("Threshold",
                       ["sensor_type", "comparison", "threshold_value", "unit", "action_text"])
def make_threshold(sensor_type, comparison, threshold_value, unit, action_text):
    return Threshold(sensor_type, comparison, threshold_value, unit, action_text)


# GPU 호환 어댑터: GpuSample → list[SensorReading]
def gpu_sample_to_readings(sample) -> list:
    """GpuSample을 SensorReading 리스트로 변환. GPU 도메인 특수화 → 범용화."""
    out = []
    ts = 0.0
    if sample.temperature_c is not None:
        out.append(make_reading(sample.name, "temperature", sample.temperature_c, "C", ts))
    if sample.power_w is not None:
        out.append(make_reading(sample.name, "power", sample.power_w, "W", ts))
    if sample.vram_used_mib is not None:
        out.append(make_reading(sample.name, "vram", sample.vram_used_mib, "MiB", ts))
    if sample.gpu_use_pct is not None:
        out.append(make_reading(sample.name, "utilization", sample.gpu_use_pct, "%", ts))
    return out


# GPU 도메인 기본 임계치 (list[Threshold])
GPU_DEFAULT_THRESHOLDS = [
    make_threshold("temperature", ">=", 40.0,  "C",   "Move quickly"),
    make_threshold("power",       ">=", 250.0, "W",   "Reduce load"),
    make_threshold("vram",        ">=", 14000.0, "MiB", "Free memory"),
]

# 배터리 도메인 기본 임계치
BATTERY_DEFAULT_THRESHOLDS = [
    make_threshold("voltage", "<=", 11.0, "V", "Return to base"),
    make_threshold("soc",     "<=", 20.0, "%", "Return to base"),
    make_threshold("current", ">=", 5.0,  "A", "Reduce load"),
]

# 모터 도메인 기본 임계치
MOTOR_DEFAULT_THRESHOLDS = [
    make_threshold("rpm",    ">=", 5000, "rpm", "Reduce load"),
    make_threshold("torque", ">=", 1.0,  "Nm",  "Reduce load"),
    make_threshold("temperature", ">=", 80.0, "C", "Stop motor"),
]

# LiDAR 도메인 기본 임계치
LIDAR_DEFAULT_THRESHOLDS = [
    make_threshold("distance", "<=", 0.5, "m", "Stop immediately"),
]

# 카메라 도메인 기본 임계치
CAMERA_DEFAULT_THRESHOLDS = [
    make_threshold("brightness", "<=", 30,   "/255", "Increase light"),
    make_threshold("contrast",   "<=", 0.1,  "ratio", "Adjust camera"),
]

# 마이크 도메인 기본 임계치
MIC_DEFAULT_THRESHOLDS = [
    make_threshold("db", ">=", 90, "dB", "Sound alarm"),
]


def derive_conditions_universal(readings, thresholds) -> dict:
    """
    [SensorReading] + [Threshold] → 파생 조건.
    반환: {"actions_by_sensor": {sensor_id: [action_text, ...]},
           "all_actions": [action_text, ...],
           "exceeded": [(reading, threshold), ...]}
    """
    actions_by_sensor = {}
    all_actions = []
    exceeded = []
    for r in readings:
        for th in thresholds:
            if r.sensor_type != th.sensor_type:
                continue
            v, t = r.value, th.threshold_value
            triggered = False
            if th.comparison == ">=" and v >= t:
                triggered = True
            elif th.comparison == "<=" and v <= t:
                triggered = True
            elif th.comparison == ">" and v > t:
                triggered = True
            elif th.comparison == "<" and v < t:
                triggered = True
            elif th.comparison == "==" and abs(v - t) < 1e-9:
                triggered = True
            if triggered:
                actions_by_sensor.setdefault(r.sensor_id, []).append(th.action_text)
                all_actions.append(th.action_text)
                exceeded.append((r, th))
    return {
        "actions_by_sensor": actions_by_sensor,
        "all_actions": all_actions,
        "exceeded": exceeded,
    }


def _declarative_clause_generic(readings) -> str:
    """
    범용 평서문: 도메인 무관 — 'The X is V.U, and the Y is V.U, and ...'
    """
    parts = []
    for r in readings:
        if r.value is None:
            continue
        # 0 또는 정수일 때 .1f로 통일 (정수형 단위 보존)
        parts.append(f"the {r.sensor_type} is {r.value:.1f}{r.unit}")
    if not parts:
        return ""
    sentence = "The " + parts[0]
    for p in parts[1:]:
        sentence += f", and {p}"
    return sentence + "."


def _conditional_clause_generic(readings, action) -> str:
    if not action or not readings:
        return ""
    r = readings[0]
    return f"If the {r.sensor_type} is {r.value:.1f}{r.unit}, {action}."


def _concessive_clause_generic(readings) -> str:
    """양보문: 'The system is stable, however, so the response remains under control.'"""
    if not readings:
        return ""
    return "The system is stable, however, so the response remains under control."

def condition_clause(sample, threshold_c: float) -> str:
    """
    조건문: "If the temperature is X°C, "
    (현재 X가 임계치 이상일 때만 호출됨)
    명령문과 같은 절로 합쳐지도록 끝에 쉼표 + 공백만 둔다.
    """
    t = sample.temperature_c
    if t is None:
        return ""   # 측정값이 없으면 발화 생략 (가짜 수치 X)
    return f"If the temperature is {t:.1f}°C,"


# ---------------------------------------------------------------------------
# 연역적 추상화: SensorReading / Threshold / compose_essay_universal
# ---------------------------------------------------------------------------
# GPU essay와 같은 3단락 구조를 *어떤 센서든* 받아들일 수 있는 *범용 함수*로
# 일반화. GPU essay는 이 함수의 *첫 검증된 인스턴스*로 남음 (기존 코드 보존).
# ---------------------------------------------------------------------------

import re as _re_generic
from dataclasses import dataclass as _dataclass_generic, field as _field_generic
from typing import List as _List, Tuple as _Tuple, Optional as _Optional


@_dataclass_generic
class SensorReading:
    """
    범용 센서 측정값. GPU / 배터리 / 모터 / LiDAR / 카메라 / 마이크
    모두 같은 형태로 표현.
    """
    sensor_id: str              # "gpu0", "battery1", "motor2", "lidar_front"
    sensor_type: str            # "temperature", "voltage", "rpm", "distance", "db"
    value: float                # 측정값
    unit: str                   # "C", "V", "m", "dB"
    timestamp: float = 0.0      # 측정 시각 (Unix epoch)
    extras: dict = _field_generic(default_factory=dict)


@_dataclass_generic
class Threshold:
    """
    임계치 + action 매핑. 도메인 무관.
    """
    sensor_type: str            # 측정값의 sensor_type과 매칭
    comparison: str             # ">=", "<=", ">", "<", "=="
    threshold_value: float
    action_text: str            # 임계치 초과/미만 시 action 문구


# 1) GpuSample → SensorReading 변환 (GPU essay 호환성용)

def gpu_sample_to_readings(sample) -> _List[SensorReading]:
    """GpuSample 1개를 SensorReading 리스트로 변환 (NA는 제외)."""
    out = []
    if sample.temperature_c is not None:
        out.append(SensorReading(
            sensor_id=f"card{sample.index}",
            sensor_type="temperature",
            value=sample.temperature_c,
            unit="C",
        ))
    if sample.power_w is not None:
        out.append(SensorReading(
            sensor_id=f"card{sample.index}",
            sensor_type="power",
            value=sample.power_w,
            unit="W",
        ))
    if sample.vram_used_mib is not None:
        out.append(SensorReading(
            sensor_id=f"card{sample.index}",
            sensor_type="vram",
            value=sample.vram_used_mib,
            unit="MiB",
        ))
    if sample.gpu_use_pct is not None:
        out.append(SensorReading(
            sensor_id=f"card{sample.index}",
            sensor_type="utilization",
            value=sample.gpu_use_pct,
            unit="%",
        ))
    return out


def gpu_thresholds(threshold_c: float = 40.0,
                    threshold_w: float = 250.0,
                    threshold_vram: float = 14000.0) -> _List[Threshold]:
    """GPU 도메인 기본 임계치."""
    return [
        Threshold("temperature", ">=", threshold_c,    "Move quickly"),
        Threshold("power",       ">=", threshold_w,    "Reduce load"),
        Threshold("vram",        ">=", threshold_vram, "Free memory"),
    ]


# 2) 범용 임계치 검사 / action 도출

def _compare_generic(value: float, comparison: str, threshold: float) -> bool:
    if comparison == ">=":
        return value >= threshold
    if comparison == "<=":
        return value <= threshold
    if comparison == ">":
        return value > threshold
    if comparison == "<":
        return value < threshold
    if comparison == "==":
        return value == threshold
    return False


def derive_conditions_universal(
    readings: _List[SensorReading],
    thresholds: _List[Threshold],
) -> dict:
    """
    readings와 thresholds를 받아 초과/미만 여부 + action 목록을 반환.
    """
    triggered = []
    actions = []
    for r in readings:
        for th in thresholds:
            if r.sensor_type != th.sensor_type:
                continue
            if _compare_generic(r.value, th.comparison, th.threshold_value):
                triggered.append((r, th, th.action_text))
                if th.action_text not in actions:
                    actions.append(th.action_text)
    return {
        "triggered": triggered,
        "actions": actions,
        "primary_action": actions[0] if actions else "",
    }


# 3) 범용 5문장 단락 합성

def _format_reading_generic(r: SensorReading) -> str:
    """단일 SensorReading을 'the power is 28.0W' 형식으로."""
    return f"the {r.sensor_type} is {r.value:.1f}{r.unit}"


def declarative_clause_generic(readings: _List[SensorReading]) -> str:
    """평서문: 'The temperature is 44.0°C, and the power is 28.0W, ...'"""
    if not readings:
        return ""
    priority = ["temperature", "voltage", "power", "rpm", "vram", "utilization",
                "distance", "brightness", "db"]
    sorted_r = sorted(readings,
                      key=lambda r: priority.index(r.sensor_type)
                                    if r.sensor_type in priority else 99)
    first = sorted_r[0]
    head = f"The {first.sensor_type} is {first.value:.1f}{first.unit}"
    if len(sorted_r) == 1:
        return head + "."
    rest = [_format_reading_generic(r) for r in sorted_r[1:]]
    return head + ", and " + ", and ".join(rest) + "."


def condition_clause_generic(
    readings: _List[SensorReading],
    action: str,
    primary_sensor: str = "temperature",
) -> str:
    """조건문: 'If the temperature is 44.0°C, '"""
    r = next((r for r in readings if r.sensor_type == primary_sensor), None)
    if r is None:
        return ""
    return f"If the {r.sensor_type} is {r.value:.1f}{r.unit},"


def imperative_clause_generic(action: str) -> str:
    """명령문: 'Move quickly.'"""
    if not action:
        return ""
    return f"{action}."


def concessive_clause_generic(
    readings: _List[SensorReading],
    primary_action: str,
) -> str:
    """양보문"""
    if primary_action:
        return (f"The system has detected a threshold breach, however, "
                f"so the response will be to {primary_action}.")
    return ("The system is stable, however, "
            "so the response remains under control.")


def compose_paragraph_universal(
    readings: _List[SensorReading],
    *,
    action: str = "",
) -> str:
    """5문장 단락 합성. readings와 action을 받아 결정적으로 단락 생성."""
    parts = []
    if action:
        c = condition_clause_generic(readings, action)
        if c:
            parts.append(c)
        i = imperative_clause_generic(action)
        if i:
            parts.append(i)
    d = declarative_clause_generic(readings)
    if d:
        parts.append(d)
    y = concessive_clause_generic(readings, action)
    if y:
        parts.append(y)
    return " ".join(parts)


# 4) 범용 3단락 essay 합성

def compose_essay_universal(
    readings: _List[SensorReading],
    thresholds: _List[Threshold],
) -> dict:
    """
    readings와 thresholds를 받아 3-Paragraph essay를 결정적으로 합성.
    반환: {"intro", "body", "conclusion", "full_essay", "action", "triggered"}
    """
    cond = derive_conditions_universal(readings, thresholds)
    action = cond["primary_action"]
    triggered = cond["triggered"]

    if readings:
        first = readings[0]
        intro = (
            f"This report describes the current state of the {first.sensor_id} sensor. "
            f"The system continuously monitors the {first.sensor_type} "
            f"to determine whether intervention is required."
        )
    else:
        intro = "This report describes the current sensor state."

    body = compose_paragraph_universal(readings, action=action)

    if action:
        conclusion = (
            f"In conclusion, the system detected the threshold breach "
            f"and responded with the action '{action}'. The monitoring continues."
        )
    else:
        conclusion = (
            "In conclusion, the system observed normal operation. "
            "The monitoring continues."
        )

    full_essay = f"{intro}\n\n{body}\n\n{conclusion}"
    return {
        "intro": intro,
        "body": body,
        "conclusion": conclusion,
        "full_essay": full_essay,
        "action": action,
        "triggered": triggered,
    }

def compose_essay_universal(readings, thresholds, domain="gpu"):
    """
    도메인 무관 essay 합성.
    domain: 'gpu', 'battery', 'motor', 'lidar', 'camera', 'mic' 등.
    """
    # 1) 도메인별 도입 단락
    intros = {
        "gpu":     "This report describes the current state of GPU thermal and power telemetry, focusing on temperature, power, VRAM, and utilization.",
        "battery": "This report describes the current state of battery health, focusing on voltage, state of charge, and current draw.",
        "motor":   "This report describes the current state of motor operation, focusing on RPM, torque, and temperature.",
        "lidar":   "This report describes the current state of LiDAR sensing, focusing on distance, reflectivity, and point cloud density.",
        "camera":  "This report describes the current state of camera vision, focusing on brightness, contrast, and object count.",
        "mic":     "This report describes the current state of microphone input, focusing on sound level, frequency, and voice detection.",
    }
    intro = intros.get(domain, "This report describes the current sensor readings.")

    # 2) 도메인별 결론 단락
    conclusions = {
        "gpu":     "The GPU requires attention to maintain thermal stability under sustained load. The monitoring continues.",
        "battery": "The battery has sufficient charge for continued operation. The monitoring continues.",
        "motor":   "The motor is operating within safe parameters. The monitoring continues.",
        "lidar":   "The LiDAR is detecting the surrounding environment. The monitoring continues.",
        "camera":  "The camera is observing the scene. The monitoring continues.",
        "mic":     "The microphone is monitoring the audio environment. The monitoring continues.",
    }
    conclusion = conclusions.get(domain, "The monitoring continues.")

    # 3) 본문 단락 — 기존 로직 (파생 조건 + 5문장 단락)
    body = ...

    return f"{intro}\n\n{body}\n\n{conclusion}"



# 5) GPU essay → generic essay 동등성 검증용 어댑터

def compose_essay_from_gpu(
    sample,
    *,
    threshold_c: float = 40.0,
    threshold_w: float = 250.0,
    threshold_vram: float = 14000.0,
) -> dict:
    """
    GpuSample 1개를 받아 compose_essay_universal()으로 essay 생성.
    기존 compose_essay(GpuSample) 결과와 비교 가능.
    """
    readings = gpu_sample_to_readings(sample)
    thresholds = gpu_thresholds(threshold_c, threshold_w, threshold_vram)
    return compose_essay_universal(readings, thresholds)


def imperative_clause(action: str) -> str:
    """
    명령문: "Move quickly."
    action은 derive_conditions()가 만든 키워드 (예: "Move quickly").
    """
    if not action:
        return ""
    return f"{action}."


def declarative_clause(sample) -> str:
    """
    평서문: "The temperature is 44.0°C."
    있는 필드만 포함, NA는 생략.
    """
    parts = []
    if sample.temperature_c is not None:
        parts.append(f"The temperature is {sample.temperature_c:.1f}°C")
    if sample.power_w is not None:
        parts.append(f"the power is {sample.power_w:.1f}W")
    if sample.vram_used_mib is not None:
        parts.append(f"the VRAM usage is {sample.vram_used_mib:.1f}MiB")
    if sample.gpu_use_pct is not None:
        parts.append(f"the GPU utilization is {sample.gpu_use_pct:.1f}%")
    if not parts:
        return ""
    # 첫 항목은 이미 "The ..." (대문자 시작), 나머지는 ", the ..."
    sentence = parts[0]
    for p in parts[1:]:
        sentence += f", and {p}"
    return sentence + "."


def concessive_clause(sample) -> str:
    """
    양보문: "The system is stable, however, so the response remains under control."
    안정성 판단:
      - 사용률 < 50% AND 전력 < 임계치 → "stable"
      - 그 외 → "under stress" 등
    """
    # 안정성 휴리스틱
    stable = True
    reasons = []
    if sample.temperature_c is not None and sample.temperature_c >= 80.0:
        stable = False
        reasons.append("high temperature")
    if sample.gpu_use_pct is not None and sample.gpu_use_pct >= 90.0:
        stable = False
        reasons.append("high utilization")

    if stable:
        return ("The system is stable, however, so the response remains under control.")
    else:
        reason_str = " and ".join(reasons)
        return (f"The system is under stress due to {reason_str}, however, "
                f"so the response must remain within safety limits.")


def compose_four_sentence_utterance(
    sample,
    *,
    threshold_c: float = 70.0,
    action: str = "",
) -> str:
    """
    4종 문장 함수를 호출해 한 단락으로 연결.
    측정값/파생 조건에 따라 일부 절이 생략될 수 있음.

    출력 예 (7900 GRE 44°C, action="Move quickly", stable):
        "If the temperature is 44.0°C, Move quickly. The temperature is 44.0°C.
         The system is stable, however, so the response remains under control."
    """
    parts = []
    if action:
        c = condition_clause(sample, threshold_c)
        if c:
            parts.append(c)
        i = imperative_clause(action)
        if i:
            parts.append(i)
    d = declarative_clause(sample)
    if d:
        parts.append(d)
    y = concessive_clause(sample)
    if y:
        parts.append(y)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 10종 복합문 생성기 (사용자 발상: 전통 문법의 모든 복합문을 함수로)
# - 센서-기반, 결정적, LLM 호출 0회
# - 모든 가능한 문장을 한 번에 생성하는 generate_all_complex_sentences() 제공
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 연역적 범용 프레임워크 (도메인 무관: GPU/배터리/모터/LiDAR/카메라/마이크)
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """범용 센서 측정값. GPU/배터리/모터/LiDAR 등 모든 도메인에 적용."""
    sensor_id: str           # "gpu0", "battery1", "motor2", "lidar_front"
    sensor_type: str         # "temperature", "voltage", "rpm", "distance", "db"
    value: float             # 측정값
    unit: str                # "C", "V", "m", "dB" 등
    timestamp: float = 0.0   # 측정 시각 (Unix epoch)
    extras: dict = field(default_factory=dict)


@dataclass
class Threshold:
    """도메인 무관 임계치 정의."""
    sensor_type: str         # SensorReading.sensor_type과 매칭
    field: str = "value"     # "value" 또는 더 세부 필드
    comparison: str = ">="   # ">=", "<=", ">", "<", "=="
    threshold_value: float = 0.0
    action_text: str = "Act" # 임계치 초과/미만 시 action 문구
    direction: str = "high"  # "high" (>=) vs "low" (<=) — action 의미 구분


def derive_conditions_universal(readings, thresholds):
    """
    readings: list[SensorReading]
    thresholds: list[Threshold]
    반환: {"any_action": bool, "all_actions": [str], "matched": list[(sensor, threshold)]}
    """
    matched = []
    all_actions = []
    for th in thresholds:
        for r in readings:
            if r.sensor_type != th.sensor_type:
                continue
            # 비교
            v = r.value
            if th.comparison == ">=" and v >= th.threshold_value:
                matched.append((r, th))
                all_actions.append(th.action_text)
            elif th.comparison == "<=" and v <= th.threshold_value:
                matched.append((r, th))
                all_actions.append(th.action_text)
            elif th.comparison == ">" and v > th.threshold_value:
                matched.append((r, th))
                all_actions.append(th.action_text)
            elif th.comparison == "<" and v < th.threshold_value:
                matched.append((r, th))
                all_actions.append(th.action_text)
            elif th.comparison == "==" and v == th.threshold_value:
                matched.append((r, th))
                all_actions.append(th.action_text)
    return {
        "any_action": bool(all_actions),
        "all_actions": all_actions,
        "matched": matched,
    }


def _declarative_clause_generic(readings) -> str:
    """
    평서문: "The temperature is 44.0°C, and the voltage is 12.4V, and ..."
    SensorReading 리스트의 모든 value를 한 문장으로 묶음.
    """
    if not readings:
        return ""
    parts = []
    for i, r in enumerate(readings):
        if i == 0:
            parts.append(f"The {r.sensor_type} is {r.value:.1f}{r.unit}")
        else:
            parts.append(f"the {r.sensor_type} is {r.value:.1f}{r.unit}")
    if len(parts) == 1:
        return parts[0] + "."
    sentence = parts[0]
    for p in parts[1:-1]:
        sentence += f", and {p}"
    sentence += f", and {parts[-1]}"
    return sentence + "."


def _conditional_clause_generic(readings, action) -> str:
    """조건문: 'If the temperature is X°C, Move quickly.' (첫 reading 기준)"""
    if not readings or not action:
        return ""
    r = readings[0]
    return f"If the {r.sensor_type} is {r.value:.1f}{r.unit}, {action}."


def _concessive_clause_generic(readings) -> str:
    """양보문: 'The system is stable, however, so the response remains under control.'"""
    if not readings:
        return ""
    # 안정성 휴리스틱 (도메인 무관): 모든 value가 0보다 크면 stable
    return "The system is stable, however, so the response remains under control."


def _comparative_clause_generic(readings, thresholds) -> str:
    """비교문: 'The temperature is higher than the threshold (X°C >= Y°C).'"""
    if not readings or not thresholds:
        return ""
    r = readings[0]
    for th in thresholds:
        if r.sensor_type == th.sensor_type:
            return f"The {r.sensor_type} is {r.value:.1f}{r.unit} (threshold: {th.comparison} {th.threshold_value:.1f}{r.unit})."
    return f"The {r.sensor_type} is {r.value:.1f}{r.unit}."


def compose_paragraph_universal(readings, thresholds) -> str:
    """
    5문장 단락 (도메인 무관):
      Topic → Detail (declarative) → Evidence (comparative) → Action → Conclusion
    """
    if not readings:
        return ""
    cond = derive_conditions_universal(readings, thresholds)
    primary_action = cond["all_actions"][0] if cond["all_actions"] else ""

    topic = "The system is monitoring sensor readings."
    detail = _declarative_clause_generic(readings)
    evidence = _comparative_clause_generic(readings, thresholds)
    action_sent = f"{primary_action}." if primary_action else ""
    conclusion = _concessive_clause_generic(readings)

    return " ".join(filter(None, [topic, detail, evidence, action_sent, conclusion]))


def compose_essay_universal(readings, thresholds) -> str:
    """
    3-Paragraph essay (도메인 무관):
      Introduction → Body (5문장 단락) → Conclusion
    """
    if not readings:
        return ""
    cond = derive_conditions_universal(readings, thresholds)
    primary_action = cond["all_actions"][0] if cond["all_actions"] else ""

    intro = (
        "This report describes the current state of sensor readings. "
        "The system continuously monitors multiple sensor channels to determine "
        "whether intervention is required."
    )
    body = compose_paragraph_universal(readings, thresholds)
    if primary_action:
        conclusion = (
            "In conclusion, the system has detected a condition requiring action. "
            f"The system will respond with '{primary_action}' and continue to "
            "monitor the sensors for any further changes."
        )
    else:
        conclusion = (
            "In conclusion, all sensor readings are within normal parameters. "
            "The system remains stable and will continue to monitor the sensors "
            "for any future changes."
        )

    return f"{intro}\n\n{body}\n\n{conclusion}"


def compose_multi_domain_essay(readings_by_domain, thresholds_by_domain) -> str:
    """
    다중 도메인 essay: GPU + 배터리 + 모터 등의 essay를 통합.
    readings_by_domain: dict[str, list[SensorReading]] — {"gpu": [...], "battery": [...]}
    thresholds_by_domain: dict[str, list[Threshold]]
    반환: 도메인별 Body 단락 + 통합 Conclusion
    """
    intro = (
        "This report describes the current state of the system across multiple "
        "sensor domains. The system continuously monitors each domain to determine "
        "whether intervention is required."
    )

    body_paragraphs = []
    all_actions = []
    for domain, readings in readings_by_domain.items():
        thresholds = thresholds_by_domain.get(domain, [])
        if readings:
            body_paragraphs.append(
                f"[{domain.upper()}] " + compose_paragraph_universal(readings, thresholds)
            )
            cond = derive_conditions_universal(readings, thresholds)
            all_actions.extend(cond["all_actions"])

    body = "\n\n".join(body_paragraphs)
    if all_actions:
        unique_actions = list(dict.fromkeys(all_actions))   # 순서 유지 중복 제거
        conclusion = (
            "In conclusion, the system has detected conditions requiring action across "
            f"the following: {unique_actions}. The system will respond accordingly "
            "and continue to monitor all sensor domains."
        )
    else:
        conclusion = (
            "In conclusion, all sensor domains are within normal parameters. "
            "The system remains stable and will continue to monitor."
        )
    return f"{intro}\n\n{body}\n\n{conclusion}"

def compound_sentence(sample, action: str = "") -> str:
    """
    1. 등위복합문: S1 + and/but + S2
    """
    d = declarative_clause(sample).rstrip(".")
    if not d:
        return ""
    if action:
        return f"{d}, and {action}."
    return f"{d}, and the system continues to monitor."


def conditional_sentence(sample, threshold_c: float, action: str = "") -> str:
    """
    2. 조건부종속복합문: If S, S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return f"If the temperature is {t:.1f}°C, {action}."
    return f"If the temperature is {t:.1f}°C, the system will respond."


def causal_sentence(sample, action: str = "") -> str:
    """
    3. 인과종속복합문: S because S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return f"{action}, because the temperature is {t:.1f}°C."
    return f"The system is monitoring the temperature, because it is {t:.1f}°C."


def temporal_sentence(sample, action: str = "") -> str:
    """
    4. 시간종속복합문: When/While S, S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return f"When the temperature reaches {t:.1f}°C, {action}."
    return f"While the temperature is {t:.1f}°C, the system remains active."


def concessive_sentence(sample, action: str = "") -> str:
    """
    5. 양보종속복합문: Although/Though S, S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return f"Although the temperature is {t:.1f}°C, the response is to {action}."
    return (f"Although the temperature is {t:.1f}°C, "
            f"the system remains stable.")

def purpose_sentence(sample, action: str = "") -> str:
    """
    6. 목적종속복합문: S so that S2
    """
    if not action:
        return ""
    t = sample.temperature_c
    if t is None:
        return f"The system will {action}, so that the workload can be managed."
    return f"The system will {action}, so that the temperature ({t:.1f}°C) can be reduced."


def result_sentence(sample, action: str = "") -> str:
    """
    7. 결과종속복합문: S, so S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return f"The temperature is {t:.1f}°C, so the system will {action}."
    return f"The temperature is {t:.1f}°C, so the system remains active."


def comparative_sentence(sample, threshold_c: float) -> str:
    """
    8. 비교종속복합문: S + than S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if t >= threshold_c:
        return f"The temperature is higher than the threshold ({t:.1f}°C >= {threshold_c:.1f}°C)."
    return f"The temperature is lower than the threshold ({t:.1f}°C < {threshold_c:.1f}°C)."


def relative_sentence(sample, action: str = "") -> str:
    """
    9. 관계종속복합문: S, which V, S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return (f"The temperature, which is {t:.1f}°C, "
                f"requires the system to {action}.")
    return (f"The temperature, which is {t:.1f}°C, "
            f"is being monitored by the system.")


def concessive_adversative_sentence(sample, action: str = "") -> str:
    """
    10. 양보-역접 복합문: Even though S, S2
    """
    t = sample.temperature_c
    if t is None:
        return ""
    if action:
        return (f"Even though the temperature is {t:.1f}°C, "
                f"the system will {action}.")
    return (f"Even though the temperature is {t:.1f}°C, "
            f"the system remains under control.")


# 모든 복합문 종류와 그 함수 매핑
COMPLEX_SENTENCE_FUNCTIONS = {
    "compound":          compound_sentence,
    "conditional":       conditional_sentence,
    "causal":            causal_sentence,
    "temporal":          temporal_sentence,
    "concessive":        concessive_sentence,
    "purpose":           purpose_sentence,
    "result":            result_sentence,
    "comparative":       comparative_sentence,
    "relative":          relative_sentence,
    "concessive_adversative": concessive_adversative_sentence,
}


def generate_all_complex_sentences(sample, *, threshold_c: float = 70.0,
                                   action: str = "") -> dict[str, str]:
    """
    10종 복합문을 *모두* 생성해 dict로 반환.
    빈 문자열인 종류는 측정값 부족으로 생성 불가.
    """
    out = {}
    for name, fn in COMPLEX_SENTENCE_FUNCTIONS.items():
        try:
            if name == "comparative":
                s = fn(sample, threshold_c)
            elif name in ("conditional",):
                s = fn(sample, threshold_c, action=action)
            else:
                s = fn(sample, action=action)
        except Exception as e:
            s = f"<error: {e}>"
        out[name] = s
    return out


# ---------------------------------------------------------------------------
# 3-Paragraph Essay 합성 (사용자 발상: 문장 → 문단 → essay)
# ---------------------------------------------------------------------------

def compose_intro_paragraph(sample, *, threshold_c: float) -> str:
    """
    Essay 단락 1: Introduction — 측정 대상과 thesis statement.
    """
    parts = []
    t = sample.temperature_c
    if t is not None:
        if t >= threshold_c:
            parts.append(
                "This report describes a GPU whose temperature has exceeded the operational threshold, "
                "requiring attention from the system."
            )
        else:
            parts.append(
                "This report describes a GPU that is currently operating within its normal parameters, "
                "with the system continuing its routine monitoring."
            )
    else:
        parts.append(
            "This report describes the current state of GPU telemetry, "
            "as observed by the monitoring system."
        )
    return " ".join(parts)


def compose_body_paragraph(sample, *, threshold_c: float, action: str = "") -> str:
    """
    Essay 단락 2: Body — compose_paragraph() 5문장 단락 재사용.
    """
    return compose_four_sentence_utterance(
        sample, threshold_c=threshold_c, action=action
    )


def compose_conclusion_paragraph(sample, *, threshold_c: float, action: str = "") -> str:
    """
    Essay 단락 3: Conclusion — 요약 + 향후 계획.
    """
    if action:
        return (
            "In conclusion, the GPU requires immediate action to maintain operational stability. "
            "The system continues to monitor the device and will respond to further changes accordingly."
        )
    return (
        "In conclusion, the GPU is operating within normal parameters. "
        "The system remains stable and will continue to monitor the device for any future changes."
    )


def compose_essay(sample, *, threshold_c: float = 70.0, action: str = "") -> str:
    """
    3-Paragraph essay 합성.
    단락 1: Introduction
    단락 2: Body (5문장 단락)
    단락 3: Conclusion
    """
    p1 = compose_intro_paragraph(sample, threshold_c=threshold_c)
    p2 = compose_body_paragraph(sample, threshold_c=threshold_c, action=action)
    p3 = compose_conclusion_paragraph(sample, threshold_c=threshold_c, action=action)
    return f"{p1}\n\n{p2}\n\n{p3}"


def compose_essay_timeseries(history, *, threshold_c: float = 70.0,
                              action: str = "") -> str:
    """
    시계열 essay 합성: 도입에 추세 요약, 본론에 4-way 비교, 결론에 추세 반영.
    history: list[list[GpuSample]] — gt.collect_recent_samples() 결과.
    """
    if not history:
        return "No telemetry data available."

    summary = gt.summarize_timeseries(history)
    latest = history[-1]
    primary = latest[0] if latest else None

    # 단락 1: Introduction + 추세
    trend_lines = []
    for idx in sorted(summary.keys()):
        s = summary[idx]
        if s["delta"] is None:
            continue
        sign = "+" if s["delta"] >= 0 else ""
        trend_lines.append(
            f"card{idx} {s['trend']} ({s['earliest']:.1f}°C → {s['current']:.1f}°C, "
            f"{sign}{s['delta']:.1f}°C)"
        )
    intro = (
        "This report describes the GPU telemetry over the recent measurement window. "
        + "Observed trends: " + "; ".join(trend_lines) + "."
    )

    # 단락 2: Body — primary sample의 compose_paragraph
    body = compose_four_sentence_utterance(
        primary, threshold_c=threshold_c, action=action
    ) if primary else ""

    # 단락 3: Conclusion
    if action:
        conclusion = (
            f"In conclusion, the system detected the threshold breach and responded with "
            f"the action '{action}'. The monitoring continues."
        )
    else:
        conclusion = (
            "In conclusion, the system observed stable operation across the measurement window. "
            "The monitoring continues."
        )

    return f"{intro}\n\n{body}\n\n{conclusion}"


# ---------------------------------------------------------------------------
# 라이브러리 모드 진입점
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def topic_sentence(sample) -> str:
    """
    1. 주제문: GPU 텔레메트리 모니터링을 도입.
    센서 상태에 따라 두 가지 변형:
      - action 있을 때 → "GPU telemetry is being monitored because the system has detected an issue."
      - action 없을 때 → "GPU telemetry is being monitored to maintain stable operation."
    """
    return "GPU telemetry is being monitored."


def detail_sentence(sample) -> str:
    """
    2. 보조문: declarative_clause() — 측정값 전개.
    """
    return declarative_clause(sample)


def evidence_sentence(sample, threshold_c: float) -> str:
    """
    3. 근거문: comparative_sentence() — 임계치와 비교.
    """
    return comparative_sentence(sample, threshold_c)


def action_sentence(action: str) -> str:
    """
    4. 액션문: imperative_clause() — action 문구.
    action이 없으면 빈 문자열.
    """
    if not action:
        return ""
    return imperative_clause(action)


def conclusion_sentence(sample) -> str:
    """
    5. 결론문: concessive_clause() — 안정성 결론.
    """
    return concessive_clause(sample)


def compose_paragraph(sample, *, threshold_c: float = 70.0,
                      action: str = "") -> str:
    """
    5가지 역할의 문장을 *하나의 문단*으로 합성.
    전통 문법의 paragraph 구조:
      Topic → Detail → Evidence → Action → Conclusion

    빈 문자열인 문장(예: action 없음)은 자동 생략.

    출력 예 (7900 GRE 44°C, action="Move quickly", stable):
        "GPU telemetry is being monitored. The temperature is 44.0°C, and the power is 28.0W,
         and the GPU utilization is 1.0%. The temperature is higher than the threshold
         (44.0°C >= 40.0°C). Move quickly. The system is stable, however, so the response
         remains under control."
    """
    parts = [
        topic_sentence(sample),
        detail_sentence(sample),
        evidence_sentence(sample, threshold_c),
        action_sentence(action),
        conclusion_sentence(sample),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 에세이 합성 (문단 → 에세이) — 전통 3-Paragraph Essay
# 단락 1: Introduction (주제 + thesis)
# 단락 2: Body (5문장 단락 — compose_paragraph)
# 단락 3: Conclusion (요약 + 의의)
# ---------------------------------------------------------------------------

def introduction_paragraph(sample, threshold_c: float, action: str) -> str:
    """
    Introduction 단락:
      - 주제 제시: GPU telemetry가 무엇인지
      - thesis statement: 현재 상태가 어떤지 (action 유무에 따라)
    """
    base = (
        "This report describes the current state of GPU telemetry. "
        "The system continuously monitors the temperature, power, VRAM usage, "
        "and GPU utilization to determine whether intervention is required."
    )
    if action:
        thesis = (
            f" Based on the current measurements, the system has detected an issue "
            f"that exceeds the configured threshold of {threshold_c:.1f}°C, "
            f"and the system will respond with the action: {action}."
        )
    else:
        thesis = (
            f" Based on the current measurements, all values remain within the "
            f"configured thresholds (temperature threshold: {threshold_c:.1f}°C), "
            f"and the system will continue to monitor the device."
        )
    return base + thesis


def conclusion_paragraph(sample, threshold_c: float, action: str) -> str:
    """
    Conclusion 단락:
      - 요약 + 의의 + 향후 계획
    """
    if action:
        return (
            "In conclusion, the GPU requires immediate action to maintain operational "
            "stability. The system has identified the issue based on the current "
            "telemetry and will continue to monitor the device for any further changes. "
            "Future measurements will be evaluated against the same thresholds, "
            "and the system will respond accordingly."
        )
    else:
        return (
            "In conclusion, the GPU is operating within normal parameters. "
            "The system remains stable and will continue to monitor the device for any "
            "future changes. The current measurements indicate no immediate action "
            "is required, and the device can continue its current workload safely."
        )


def compose_essay(sample, *, threshold_c: float = 70.0,
                  action: str = "") -> str:
    """
    3-Paragraph Essay 합성.
      - 단락 1: Introduction (주제 + thesis)
      - 단락 2: Body (compose_paragraph — 5문장 단락)
      - 단락 3: Conclusion (요약 + 의의)

    각 단락은 빈 줄(\\n\\n)로 구분. 전통 3-Paragraph Essay 구조.
    """
    intro = introduction_paragraph(sample, threshold_c, action)
    body = compose_paragraph(sample, threshold_c=threshold_c, action=action)
    conclusion = conclusion_paragraph(sample, threshold_c, action)
    return f"{intro}\n\n{body}\n\n{conclusion}"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def run_once(
    target_indices=(0, 1, 2, 3),
    *,
    thresholds=None,
    url: str = DEFAULT_LLAMA_URL,
    model: str = DEFAULT_MODEL,
    timeout = DEFAULT_TIMEOUT,
    mode: str = "qwen",   # "qwen" (LLM 호출) | "four-sentence" (4종 문장 함수)
                          # | "all-complex" (10종 복합문) | "paragraph" (5문장 단락)
                          # | "essay-timeseries" (시계열 essay)
    n_samples: int = 3,
    interval_sec: float = 1.0,
) -> GroundingAttempt:
    """
    라이브 한 사이클:
      - mode="qwen": 텔레메트리 → 파생 조건 → Qwen → 결과
      - mode="four-sentence": 텔레메트리 → 파생 조건 → 4종 문장 함수 → 결과
                              (LLM 호출 없음, 결정적)
      - mode="all-complex": 텔레메트리 → 10종 복합문 모두 생성 (LLM 호출 없음, 결정적)
      - mode="paragraph": 텔레메트리 → 5가지 역할의 전통 문단 (LLM 호출 없음, 결정적)
    thresholds: None이면 DEFAULT_THRESHOLDS 사용.
    """
    samples = gt.collect(target_indices=target_indices)
    vec = gt.to_vector_string(samples)
    cond = derive_conditions(samples, thresholds=thresholds)
    cond_text = conditions_to_prompt_text(cond)

    if mode == "four-sentence":
        # 4종 문장 함수로 단락 생성 — LLM 호출 없음, 결정적
        # 첫 번째 카드의 action만 사용 (간단화)
        primary_sample = samples[0] if samples else None
        primary_action = cond["all_actions"][0] if cond["all_actions"] else ""
        primary_threshold = (thresholds or DEFAULT_THRESHOLDS)["temp_high_c"]
        en = (compose_four_sentence_utterance(
                  primary_sample,
                  threshold_c=primary_threshold,
                  action=primary_action,
              )
              if primary_sample else "")
        # 한국어 버전은 직전 라운드의 Qwen KO 발화 형식을 모방하는 정직한 평서문
        if primary_sample is not None:
            t = primary_sample.temperature_c
            p = primary_sample.power_w
            ko = (f"card0의 GPU 온도는 {t:.1f}도" if t is not None else "card0의 GPU 온도는 N/A")
            if p is not None:
                ko += f", 전력은 {p:.1f}W"
            ko += "이다."
            if primary_action:
                ko += f" 임계치 {primary_threshold:.1f}도 이상이므로 {primary_action}가 필요하다."
        else:
            ko = ""
        return GroundingAttempt(
            vector_input=vec,
            conditions_text=cond_text,
            thresholds=cond["thresholds_used"],
            raw_response=f"[4-sentence mode, no LLM call]\nEN: {en}\nKO: {ko}",
            ko_sentence=ko,
            en_sentence=en,
            http_status=200,   # 결정적이므로 200 OK처럼 표시
        )

    if mode == "essay-timeseries":
        targets = target_indices
        # 시계열 수집 + essay 합성
        from gpu_telemetry import collect_recent_samples
        history = collect_recent_samples(
            n_samples=n_samples,
            interval_sec=interval_sec,
            target_indices=targets,
            timeout=2.0,
        )
        summary = gt.summarize_timeseries(history)
        primary_action = cond["all_actions"][0] if cond["all_actions"] else ""
        primary_threshold = (thresholds or DEFAULT_THRESHOLDS)["temp_high_c"]
        primary_sample = (history[-1][0] if history and history[-1] else None)
        essay = (compose_essay_timeseries(
                     history,
                     threshold_c=primary_threshold,
                     action=primary_action,
                 ) if primary_sample else "No telemetry data.")
        # KO는 단일 평서문 (현 시점의 primary sample 기준)
        if primary_sample is not None:
            t = primary_sample.temperature_c
            p = primary_sample.power_w
            ko = (f"card0의 GPU 온도는 {t:.1f}도" if t is not None else "card0의 GPU 온도는 N/A")
            if p is not None:
                ko += f", 전력은 {p:.1f}W"
            ko += "이다."
            if primary_action:
                ko += f" 임계치 {primary_threshold:.1f}도 이상이므로 {primary_action}가 필요하다."
        else:
            ko = ""
        return GroundingAttempt(
            vector_input=vec,
            conditions_text=cond_text,
            thresholds=cond["thresholds_used"],
            raw_response=f"[essay-timeseries mode, no LLM call]\n{essay}",
            ko_sentence=ko,
            en_sentence=essay,
            http_status=200,
        )

    if mode == "all-complex":
        # 10종 복합문 함수로 단락 모두 생성 — LLM 호출 없음, 결정적
        primary_sample = samples[0] if samples else None
        primary_action = cond["all_actions"][0] if cond["all_actions"] else ""
        primary_threshold = (thresholds or DEFAULT_THRESHOLDS)["temp_high_c"]
        all_sentences = (generate_all_complex_sentences(
                            primary_sample,
                            threshold_c=primary_threshold,
                            action=primary_action,
                         ) if primary_sample else {})
        en = "\n".join(f"[{k}] {v}" for k, v in all_sentences.items() if v)
        ko = (f"card0의 GPU 온도는 {primary_sample.temperature_c:.1f}도"
              if (primary_sample and primary_sample.temperature_c is not None)
              else "card0의 GPU 온도는 N/A")
        if primary_sample and primary_sample.power_w is not None:
            ko += f", 전력은 {primary_sample.power_w:.1f}W"
        ko += "이다."
        if primary_action:
            ko += f" 임계치 {primary_threshold:.1f}도 이상이므로 {primary_action}가 필요하다."
        return GroundingAttempt(
            vector_input=vec,
            conditions_text=cond_text,
            thresholds=cond["thresholds_used"],
            raw_response=f"[all-complex mode, no LLM call]\n{en}",
            ko_sentence=ko,
            en_sentence=en,
            http_status=200,
        )

    if mode == "essay":
        # 3-Paragraph Essay 합성 — LLM 호출 없음, 결정적
        primary_sample = samples[0] if samples else None
        primary_action = cond["all_actions"][0] if cond["all_actions"] else ""
        primary_threshold = (thresholds or DEFAULT_THRESHOLDS)["temp_high_c"]
        en = (compose_essay(
                  primary_sample,
                  threshold_c=primary_threshold,
                  action=primary_action,
              )
              if primary_sample else "")
        # 한국어 essay: 3단락
        if primary_sample is not None:
            t = primary_sample.temperature_c
            p = primary_sample.power_w
            u = primary_sample.gpu_use_pct
            # 단락 1: 도입
            ko_intro = (
                "이 보고서는 GPU 텔레메트리의 현재 상태를 설명한다. "
                "시스템은 온도, 전력, VRAM, 사용률을 지속적으로 모니터링하여 "
                "조치가 필요한지 판단한다."
            )
            if primary_action:
                ko_intro += (
                    f" 현재 측정값에 따르면 시스템은 임계치 {primary_threshold:.1f}도를 "
                    f"초과하는 문제를 감지했으며, 다음과 같이 조치한다: {primary_action}."
                )
            else:
                ko_intro += (
                    f" 현재 측정값에 따르면 모든 값이 설정된 임계치 이내이다 "
                    f"(온도 임계치: {primary_threshold:.1f}도). 시스템은 계속 모니터링한다."
                )
            # 단락 2: 본론
            ko_body_parts = ["GPU 텔레메트리가 모니터링되고 있다."]
            if t is not None:
                detail_ko = f"현재 card0의 GPU 온도는 {t:.1f}도"
                if p is not None:
                    detail_ko += f", 전력은 {p:.1f}W"
                if u is not None:
                    detail_ko += f", 사용률은 {u:.1f}%"
                detail_ko += "이다."
                ko_body_parts.append(detail_ko)
                if t >= primary_threshold:
                    ko_body_parts.append(
                        f"이 온도는 임계치 {primary_threshold:.1f}도 이상이다."
                    )
                else:
                    ko_body_parts.append(
                        f"이 온도는 임계치 {primary_threshold:.1f}도 미만이다."
                    )
            if primary_action:
                ko_body_parts.append(f"{primary_action}가 필요하다.")
            if t is not None and t < 80.0 and (u is None or u < 90.0):
                ko_body_parts.append("시스템은 안정적으로 제어되고 있다.")
            else:
                ko_body_parts.append("시스템은 부하 상태이지만, 응답은 안전 한계 내에서 유지된다.")
            ko_body = " ".join(ko_body_parts)
            # 단락 3: 결론
            if primary_action:
                ko_conclusion = (
                    "결론적으로, GPU는 운영 안정성을 유지하기 위해 즉각적인 조치가 필요하다. "
                    "시스템은 현재 텔레메트리 기반으로 문제를 식별했으며, "
                    "추가 변화가 있는지 계속 모니터링할 것이다. "
                    "향후 측정값도 동일한 임계치로 평가되며, "
                    "시스템은 그에 따라 응답할 것이다."
                )
            else:
                ko_conclusion = (
                    "결론적으로, GPU는 정상 매개변수 내에서 작동하고 있다. "
                    "시스템은 안정 상태를 유지하며, 향후 변화가 있는지 계속 모니터링할 것이다. "
                    "현재 측정값은 즉각적인 조치가 필요하지 않음을 나타내며, "
                    "장치는 현재 작업을 안전하게 계속할 수 있다."
                )
            ko = f"{ko_intro}\n\n{ko_body}\n\n{ko_conclusion}"
        else:
            ko = ""
        return GroundingAttempt(
            vector_input=vec,
            conditions_text=cond_text,
            thresholds=cond["thresholds_used"],
            raw_response=f"[essay mode, no LLM call]\nEN:\n{en}",
            ko_sentence=ko,
            en_sentence=en,
            http_status=200,
        )

    if mode == "paragraph":
        # 5가지 역할의 문장으로 전통 문법 paragraph 합성 — LLM 호출 없음, 결정적
        primary_sample = samples[0] if samples else None
        primary_action = cond["all_actions"][0] if cond["all_actions"] else ""
        primary_threshold = (thresholds or DEFAULT_THRESHOLDS)["temp_high_c"]
        en = (compose_paragraph(
                  primary_sample,
                  threshold_c=primary_threshold,
                  action=primary_action,
              )
              if primary_sample else "")
        # 한국어 paragraph: 5가지 역할의 한국어 버전
        if primary_sample is not None:
            t = primary_sample.temperature_c
            p = primary_sample.power_w
            u = primary_sample.gpu_use_pct
            ko_parts = ["GPU 텔레메트리가 모니터링되고 있다."]
            if t is not None:
                detail_ko = f"현재 card0의 GPU 온도는 {t:.1f}도"
                if p is not None:
                    detail_ko += f", 전력은 {p:.1f}W"
                if u is not None:
                    detail_ko += f", 사용률은 {u:.1f}%"
                detail_ko += "이다."
                ko_parts.append(detail_ko)
                if t >= primary_threshold:
                    ko_parts.append(
                        f"이 온도는 임계치 {primary_threshold:.1f}도 이상이다."
                    )
                else:
                    ko_parts.append(
                        f"이 온도는 임계치 {primary_threshold:.1f}도 미만이다."
                    )
            if primary_action:
                ko_parts.append(f"{primary_action}가 필요하다.")
            # 결론
            if t is not None and t < 80.0 and (u is None or u < 90.0):
                ko_parts.append("시스템은 안정적으로 제어되고 있다.")
            else:
                ko_parts.append("시스템은 부하 상태이지만, 응답은 안전 한계 내에서 유지된다.")
            ko = " ".join(ko_parts)
        else:
            ko = ""
        return GroundingAttempt(
            vector_input=vec,
            conditions_text=cond_text,
            thresholds=cond["thresholds_used"],
            raw_response=f"[paragraph mode, no LLM call]\nEN: {en}\nKO: {ko}",
            ko_sentence=ko,
            en_sentence=en,
            http_status=200,
        )

    # mode == "qwen" (default)
    result = call_qwen(vec, conditions_text=cond_text, url=url, model=model, timeout=timeout)
    result.thresholds = cond["thresholds_used"]
    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="GPU 텔레메트리 그라운딩 파이프라인 — 1장 또는 4장"
    )
    p.add_argument(
        "--device", "-d", type=int, default=None,
        help="단일 디바이스 인덱스 (예: 0). 생략하면 0,1,2,3 4장 모두.",
    )
    p.add_argument(
        "--with-check", action="store_true",
        help="결과에 grounding_check 자동 검사까지 수행",
    )
    p.add_argument(
        "--save", type=str, default=None,
        help="원시 결과를 JSON 파일로 저장 (경로 지정)",
    )
    p.add_argument(
        "--threshold-temp", type=float, default=None,
        help=f"온도 임계치 (°C). 기본 {DEFAULT_THRESHOLDS['temp_high_c']:.1f}.",
    )
    p.add_argument(
        "--threshold-power", type=float, default=None,
        help=f"전력 임계치 (W). 기본 {DEFAULT_THRESHOLDS['power_high_w']:.1f}.",
    )
    p.add_argument(
        "--threshold-vram", type=float, default=None,
        help=f"VRAM 임계치 (MiB). 기본 {DEFAULT_THRESHOLDS['vram_high_mib']:.1f}.",
    )
    p.add_argument(
        "--mode", choices=["qwen", "four-sentence", "all-complex", "paragraph", "essay-timeseries"], default="qwen",
        help="발화 모드. 'qwen'=LLM 호출(기본), 'four-sentence'=4종 문장 함수, "
             "'all-complex'=10종 복합문 모두 생성, 'paragraph'=5문장 단락, "
             "'essay-timeseries'=시계열 3단락 essay.",
    )
    p.add_argument(
        "--samples", type=int, default=3,
        help="essay-timeseries 모드에서 시계열 샘플 수 (기본 3).",
    )
    p.add_argument(
        "--interval", type=float, default=1.0,
        help="essay-timeseries 모드에서 샘플 간 간격 (초, 기본 1.0).",
    )
    args = p.parse_args()

    if args.device is not None:
        targets = (args.device,)
    else:
        targets = (0, 1, 2, 3)

    # thresholds dict 구성 (None인 키는 기본값 유지)
    th = dict(DEFAULT_THRESHOLDS)
    if args.threshold_temp is not None:
        th["temp_high_c"] = args.threshold_temp
    if args.threshold_power is not None:
        th["power_high_w"] = args.threshold_power
    if args.threshold_vram is not None:
        th["vram_high_mib"] = args.threshold_vram

    res = run_once(target_indices=targets, thresholds=th, mode=args.mode,
                   n_samples=args.samples, interval_sec=args.interval)
    data = res.to_dict()
    print("=== 입력 텔레메트리 벡터 ===")
    print(data["vector_input"])
    print()
    print("=== 사용된 임계치 ===")
    for k, v in data["thresholds"].items():
        print(f"  {k} = {v}")
    print()
    print("=== 파생 조건 + action 문구 ===")
    print(data["conditions_text"])
    print()
    print("=== Qwen 원본 응답 ===")
    print(data["raw_response"])
    print()
    print("=== 추출된 KO/EN ===")
    print(f"KO: {data['ko_sentence']}")
    print(f"EN: {data['en_sentence']}")
    print()
    print(f"HTTP status: {data['http_status']}")

    if args.with_check:
        # 라이브 결과에 대해 그라운딩 자동 검사
        import grounding_check as gc
        # 검사 대상은 벡터 + 파생 조건의 action 문구도 포함
        r = gc.check_grounding(
            data["vector_input"],
            data["ko_sentence"],
            data["en_sentence"],
        )
        print()
        print("=== 그라운딩 검사 ===")
        print(r.summary())

        # action 문구가 KO/EN에 정확히 포함됐는지도 검사
        import re as _re
        action_keywords = _re.findall(r"'(Move quickly|Reduce load|Free memory)'",
                                       data["conditions_text"])
        if action_keywords:
            print()
            print("=== Action 문구 포함 검사 ===")
            for act in set(action_keywords):
                in_ko = act in data["ko_sentence"]
                in_en = act in data["en_sentence"]
                status = "✓" if (in_ko and in_en) else "✗"
                print(f"  {status} '{act}': KO={in_ko}, EN={in_en}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {args.save}")











































# ---------------------------------------------------------------------------
# 연역적 일반화: compose_essay_universal()
# - GpuSample/SensorReading 둘 다 받음
# - 도메인 무관 3-Paragraph essay 합성
# - GPU essay는 SensorReading의 인스턴스로 표현
# ---------------------------------------------------------------------------

def _reading_field(r):
    """SensorReading 또는 GpuSample에서 (value, unit, stype) 추출."""
    if hasattr(r, "value") and hasattr(r, "unit") and hasattr(r, "sensor_type"):
        return r.value, r.unit, r.sensor_type
    if hasattr(r, "temperature_c") and r.temperature_c is not None:
        return r.temperature_c, "C", "temperature"
    return None, None, None


def _check_threshold(value, threshold) -> bool:
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


def derive_conditions_universal(readings, thresholds):
    """readings × thresholds 비교 → action flags."""
    actions = {}
    for r in readings:
        value, unit, stype = _reading_field(r)
        rid = getattr(r, "sensor_id", None) or getattr(r, "name", "unknown")
        for th in thresholds:
            if th.sensor_type != stype:
                continue
            if _check_threshold(value, th):
                actions.setdefault(rid, []).append(th.action_text)
    return actions


def _format_value_with_unit(value, unit) -> str:
    if unit == "C":
        return f"{value:.1f}°C"
    if unit == "V":
        return f"{value:.1f}V"
    if unit == "W":
        return f"{value:.1f}W"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "MiB":
        return f"{value:.1f}MiB"
    if unit == "m":
        return f"{value:.1f}m"
    return f"{value:.1f}{unit}"


def compose_paragraph_universal(readings, thresholds) -> str:
    """5문장 단락 (도메인 무관)."""
    conditions = derive_conditions_universal(readings, thresholds)
    all_actions = [a for acts in conditions.values() for a in acts]
    parts = ["The system is monitoring the current sensor readings."]

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
                    parts.append(
                        f"The {stype} of {_format_value_with_unit(value, unit)} "
                        f"is higher than the threshold of {th.threshold_value:.1f}{unit}."
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
                    parts.append(
                        f"The {stype} of {_format_value_with_unit(value, unit)} "
                        f"is below the threshold of {th.threshold_value:.1f}{unit}."
                    )
                    evidence_done = True
                    break

    if all_actions:
        parts.append("The system has detected the condition and will respond accordingly.")
    else:
        parts.append("The system is stable and continues to monitor.")

    return " ".join(parts)


def compose_essay_universal(readings, thresholds) -> str:
    """3-Paragraph essay (도메인 무관)."""
    rid_set = sorted({getattr(r, "sensor_id", None) or getattr(r, "name", "unknown")
                      for r in readings})
    intro = (
        f"This report describes the current sensor readings from {len(rid_set)} "
        f"sensor(s) ({', '.join(rid_set)}). The system continuously monitors "
        f"the physical state to determine whether intervention is required."
    )
    body = compose_paragraph_universal(readings, thresholds)
    conditions = derive_conditions_universal(readings, thresholds)
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

#{body}

#{conclusion}"



































































































































