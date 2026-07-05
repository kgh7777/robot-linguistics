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
from dataclasses import dataclass

import requests

import gpu_telemetry as gt

log = logging.getLogger("grounding_pipeline")

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:2242/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "qwen2.5-32b-instruct-q8_0")
DEFAULT_TIMEOUT = (5, 300)  # (connect, read) — read는 60초까지


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
# 라이브러리 모드 진입점
# ---------------------------------------------------------------------------

def run_once(
    target_indices=(0, 1, 2, 3),
    *,
    thresholds=None,
    url: str = DEFAULT_LLAMA_URL,
    model: str = DEFAULT_MODEL,
    timeout = DEFAULT_TIMEOUT,
) -> GroundingAttempt:
    """
    라이브 한 사이클: 텔레메트리 → 파생 조건 → Qwen → 결과
    thresholds: None이면 DEFAULT_THRESHOLDS 사용.
    """
    samples = gt.collect(target_indices=target_indices)
    vec = gt.to_vector_string(samples)
    cond = derive_conditions(samples, thresholds=thresholds)
    cond_text = conditions_to_prompt_text(cond)
    result = call_qwen(vec, conditions_text=cond_text, url=url, model=model, timeout=timeout)
    # 사용된 임계치도 채워서 반환
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

    res = run_once(target_indices=targets, thresholds=th)
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






