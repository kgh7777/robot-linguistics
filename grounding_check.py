"""
grounding_check.py

모델이 인용한 수치가 입력 텔레메트리 벡터의 값과 일치하는지 자동 검사.

원리:
  - 입력 벡터 (예: "[card0 T:44.0C,P:186.0W,V:10240.0MiB,U:62.0%] ...") 에서
    카드별 (T, P, V, U) 수치를 추출
  - 모델 출력(한국어/영어)에서도 부동소수점 수치를 모두 추출
  - 입력의 *모든* 수치가 모델 출력에 *그대로* 등장하면 그라운딩 성공
  - 등장하지 않은 수치가 있으면 그라운딩 실패 (분포 모방/허구 발생)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

import gpu_telemetry as gt

log = logging.getLogger("grounding_check")

# ---------------------------------------------------------------------------
# 입력 벡터 파싱
# ---------------------------------------------------------------------------

# 예: [card0 T:44.0C,P:186.0W,V:10240.0MiB,U:62.0%]
# NA 값도 매칭해야 한다 (예: T:NAC, P:NAW, V:NAMiB, U:NA%)
_VECTOR_RE = re.compile(
    r"\[(?P<name>\S+)\s+"
    r"T:(?P<T>[\d.]+|NA)C,"
    r"P:(?P<P>[\d.]+|NA)W,"
    r"V:(?P<V>[\d.]+|NA)MiB,"
    r"U:(?P<U>[\d.]+|NA)%\]"
)

# 입력 벡터에서 NA 마커 감지 (T:NAC, P:NAW, V:NAMiB, U:NA% 모두 매칭)
_NA_MARKER_RE = re.compile(r"^NA$")


@dataclass
class VectorNumber:
    """입력 벡터에서 추출한 (카드, 필드, 값) 한 개."""
    card: str
    field: str
    raw: str  # 원문 (예: "44.0")
    is_na: bool


def parse_vector(vector_str: str) -> list[VectorNumber]:
    """
    텔레메트리 벡터 문자열에서 모든 (card, field, value) 추출.
    NA 마커는 is_na=True로 표시 (검사 제외 대상).
    """
    out: list[VectorNumber] = []
    for m in _VECTOR_RE.finditer(vector_str):
        name = m.group("name")
        for field_name in ("T", "P", "V", "U"):
            raw = m.group(field_name)
            out.append(VectorNumber(
                card=name, field=field_name, raw=raw,
                is_na=_NA_MARKER_RE.match(raw) is not None,
            ))
    return out


# ---------------------------------------------------------------------------
# 모델 출력에서 수치 추출
# ---------------------------------------------------------------------------

# 한국어/영어 문장에서 부동소수점 수치만 추출
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 1자리 정수 0~3은 GPU 디바이스 인덱스로 사용될 가능성 높음 → 환각 검사에서 제외
_DEVICE_INDEX = {"0", "1", "2", "3"}


def extract_numbers_in_text(text: str) -> list[str]:
    """
    텍스트에서 부동소수점 수치들을 등장 순서대로 모두 추출.
    '10240' 같은 정수도 '10240.0'으로 정규화되지 않은 채 등장하면 그대로 잡힘.
    """
    return _NUMBER_RE.findall(text)


def _is_likely_device_index(num_str: str) -> bool:
    """0~3 사이의 1자리 정수면 디바이스 인덱스 가능성 높음."""
    return num_str in _DEVICE_INDEX


# ---------------------------------------------------------------------------
# 그라운딩 검사 결과
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    vector_input: str
    ko_sentence: str
    en_sentence: str
    n_total: int = 0                # 입력에 있던 NA가 아닌 수치 총 개수
    n_covered_ko: int = 0           # 한국어 문장에서 등장한 입력 수치 개수
    n_covered_en: int = 0           # 영어 문장에서 등장한 입력 수치 개수
    n_covered_both: int = 0         # KO·EN 양쪽 모두 등장한 입력 수치 개수
    missing_values: list[str] = field(default_factory=list)   # 등장하지 않은 입력 수치
    hallucinated_numbers_ko: list[str] = field(default_factory=list)  # KO에 등장했지만 입력엔 없는 수치
    hallucinated_numbers_en: list[str] = field(default_factory=list)

    @property
    def ko_coverage(self) -> float:
        return (self.n_covered_ko / self.n_total) if self.n_total else 0.0

    @property
    def en_coverage(self) -> float:
        return (self.n_covered_en / self.n_total) if self.n_total else 0.0

    @property
    def both_coverage(self) -> float:
        return (self.n_covered_both / self.n_total) if self.n_total else 0.0

    @property
    def is_grounded(self) -> bool:
        """그라운딩 성공 = NA가 아닌 모든 입력 수치가 KO·EN 양쪽 모두 등장.
        단, 검사 대상 수치가 0개면 검사 무효(검사 불가)."""
        return self.n_total > 0 and self.n_covered_both == self.n_total

    @property
    def is_checkable(self) -> bool:
        """검사 대상 수치가 1개 이상일 때만 의미 있는 검사."""
        return self.n_total > 0

    def summary(self) -> str:
        checkable = "✓" if self.is_checkable else "✗ (검사 무효)"
        grounded = "✓" if self.is_grounded else "✗"
        return (
            f"그라운딩 검사 결과\n"
            f"  입력 수치(NA 제외): {self.n_total}개 (검사 가능: {checkable})\n"
            f"  KO 등장: {self.n_covered_ko} ({self.ko_coverage*100:.1f}%)\n"
            f"  EN 등장: {self.n_covered_en} ({self.en_coverage*100:.1f}%)\n"
            f"  KO·EN 모두 등장: {self.n_covered_both} ({self.both_coverage*100:.1f}%)\n"
            f"  미등장: {self.missing_values}\n"
            f"  KO 환각 수치(입력엔 없음): {self.hallucinated_numbers_ko}\n"
            f"  EN 환각 수치(입력엔 없음): {self.hallucinated_numbers_en}\n"
            f"  그라운딩 성공 여부: {grounded}\n"
        )


# ---------------------------------------------------------------------------
# 수치 매칭 로직
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """수치 문자열 정규화: '44.0'/'44'/'44.00' 모두 '44.0'으로 비교."""
    try:
        f = float(s)
        if f.is_integer():
            return f"{f:.1f}"
        return f"{f:g}"
    except ValueError:
        return s


def _covered(target: str, haystack: Iterable[str]) -> bool:
    """target이 haystack에 등장하는지 (정규화 후)."""
    t = _norm(target)
    for h in haystack:
        if _norm(h) == t:
            return True
    return False


# ---------------------------------------------------------------------------
# 메인 검사 함수
# ---------------------------------------------------------------------------

def check_grounding(
    vector_input: str,
    ko_sentence: str,
    en_sentence: str,
) -> CheckResult:
    """
    1) 입력 벡터에서 모든 수치 추출
    2) KO·EN 문장에서 모든 수치 추출
    3) NA 제외한 입력 수치가 KO·EN에 등장하는지 검사
    4) KO·EN에 등장한 수치 중 입력에 없는 것 = 환각
    """
    inputs = parse_vector(vector_input)
    non_na = [v for v in inputs if not v.is_na]

    ko_nums = extract_numbers_in_text(ko_sentence)
    en_nums = extract_numbers_in_text(en_sentence)

    # 입력 수치 등장 여부
    covered_ko = 0
    covered_en = 0
    covered_both = 0
    missing: list[str] = []

    for v in non_na:
        in_ko = _covered(v.raw, ko_nums)
        in_en = _covered(v.raw, en_nums)
        if in_ko:
            covered_ko += 1
        if in_en:
            covered_en += 1
        if in_ko and in_en:
            covered_both += 1
        if not (in_ko and in_en):
            missing.append(f"{v.card}.{v.field}={v.raw}")

    # 입력 정규화 집합
    input_set = {_norm(v.raw) for v in non_na}

    # 환각 수치 (디바이스 인덱스로 보이는 1자리 정수 0~3 제외)
    hallu_ko = [n for n in ko_nums
                if _norm(n) not in input_set and not _is_likely_device_index(n)]
    hallu_en = [n for n in en_nums
                if _norm(n) not in input_set and not _is_likely_device_index(n)]

    return CheckResult(
        vector_input=vector_input,
        ko_sentence=ko_sentence,
        en_sentence=en_sentence,
        n_total=len(non_na),
        n_covered_ko=covered_ko,
        n_covered_en=covered_en,
        n_covered_both=covered_both,
        missing_values=missing,
        hallucinated_numbers_ko=hallu_ko,
        hallucinated_numbers_en=hallu_en,
    )


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        # 데모: gpu_telemetry.collect() 결과를 벡터로 만들어 빈 문장 검사
        samples = gt.collect()
        vec = gt.to_vector_string(samples)
        print(f"[demo] vector = {vec}")
        result = check_grounding(vec, ko_sentence="", en_sentence="")
        print(result.summary())
    else:
        # stdin JSON: {"vector":..., "ko":..., "en":...}
        payload = json.loads(sys.stdin.read())
        result = check_grounding(
            payload["vector"],
            ko_sentence=payload.get("ko", ""),
            en_sentence=payload.get("en", ""),
        )
        print(result.summary())







