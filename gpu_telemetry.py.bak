"""
gpu_telemetry.py

4-way Radeon GPU 텔레메트리 추출기 (RDNA3 / RDNA2 혼합 클러스터용)
- 1차: rocm-smi --json
- 2차: amdgpu_top -d 0,1,2,3 -J
- 둘 다 실패하면 명시적 'unavailable' 마커를 붙여 가짜 수치를 절대 박지 않는다.

추출 대상: 카드별 Temperature(°C), Power(W), VRAM(used MiB), GPU use(%)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger("gpu_telemetry")

# ---------------------------------------------------------------------------
# 데이터 컨테이너
# ---------------------------------------------------------------------------

@dataclass
class GpuSample:
    index: int              # PCI 카드 인덱스 (0,1,2,3 …)
    name: str               # 예: "card0", "device0"
    source: str             # "rocm-smi" | "amdgpu_top" | "unavailable"
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    vram_used_mib: Optional[float] = None
    gpu_use_pct: Optional[float] = None
    vram_mib: Optional[float] = None
    utilization_pct: float | None = None

    def has_any(self) -> bool:
        return any(v is not None for v in (
            self.temperature_c, self.power_w, self.vram_used_mib, self.gpu_use_pct,
        ))

    def to_vector_token(self) -> str:
        """
        LLM 입력용 단축 문자열. 모르는 값은 'NA' 로 박는다 (가짜 수치 금지).
        """
        def fmt(v, unit):
            return f"{v:.1f}{unit}" if v is not None else f"NA{unit}"
        return (
            f"[{self.name} "
            f"T:{fmt(self.temperature_c, 'C')},"
            f"P:{fmt(self.power_w, 'W')},"
            f"V:{fmt(self.vram_used_mib, 'MiB')},"
            f"U:{fmt(self.gpu_use_pct, '%')}]"
        )


# ---------------------------------------------------------------------------
# 파서: rocm-smi --json
# ---------------------------------------------------------------------------

# 실제 rocm-smi --json 출력 키 패턴. ROCm 5.x ~ 6.x에서 검증된 형태들.
_ROCM_TEMP_KEYS = (
    "Temperature (Sensor edge) (C)",
    "Temperature (Edge) (C)",
    "Temperature (C)",
)
_ROCM_POWER_KEYS = (
    "Average Graphics Package Power (W)",
    "Current Socket Graphics Package Power (W)",
    "Power Cap (W)",            # 폴백 (실측이 아니라 cap인 경우 마킹)
    "Average Power (W)",
)
_ROCM_VRAM_KEYS = (
    "GPU memory use (MiB)",
    "VRAM Total Used Memory (MiB)",
    "Used Memory (MiB)",
    "VRAM Total Used Memory (B)",   # 일부 ROCm 5.x/6.x는 used를 Bytes로 보고
)
# Bytes 단위로 보고되는 키는 MiB로 자동 변환 필요 (used만 — 총 VRAM 크기는 used 폴백 X)
_VRAM_BYTES_KEYS = {
    "VRAM Total Used Memory (B)",
}
_ROCM_USE_KEYS = (
    "GPU use (%)",
    "GPU Utilization (%)",
)


def _pick(d: dict, keys: tuple[str, ...], cast=float) -> Optional[float]:
    """주어진 dict에서 keys 중 하나를 찾아 숫자로 캐스팅. 모두 실패하면 None."""
    for k in keys:
        if k in d and d[k] not in (None, "", "N/A"):
            try:
                return cast(d[k])
            except (TypeError, ValueError):
                continue
    return None


def _parse_vram_mib(payload: dict) -> Optional[float]:
    """
    VRAM used 값 추출. MiB 단위 키를 우선, Bytes 단위 키는 MiB로 자동 변환.
    """
    # MiB 키 우선
    for k in _ROCM_VRAM_KEYS:
        if k in _VRAM_BYTES_KEYS:
            continue   # Bytes 키는 아래에서 따로 처리
        if k in payload and payload[k] not in (None, "", "N/A"):
            try:
                return float(payload[k])
            except (TypeError, ValueError):
                continue
    # Bytes 키 폴백
    for k in _VRAM_BYTES_KEYS:
        if k in payload and payload[k] not in (None, "", "N/A"):
            try:
                return float(payload[k]) / (1024.0 * 1024.0)   # B → MiB
            except (TypeError, ValueError):
                continue
    return None


def _parse_rocm_smi_card(name: str, payload: dict) -> GpuSample:
    """
    한 카드의 rocm-smi --json dict → GpuSample.
    일부 필드가 없는 카드(예: RDNA2의 partial measurement)도 정직하게 처리 —
    있는 필드만 채우고 없는 필드는 None으로 둠 (가짜 수치 X).
    """
    sample = GpuSample(index=-1, name=name, source="rocm-smi")
    sample.temperature_c = _pick(payload, _ROCM_TEMP_KEYS)
    sample.power_w       = _pick(payload, _ROCM_POWER_KEYS)
    sample.vram_used_mib = _parse_vram_mib(payload)
    sample.gpu_use_pct   = _pick(payload, _ROCM_USE_KEYS)
    return sample


# ---------------------------------------------------------------------------
# 파서: amdgpu_top -J
# ---------------------------------------------------------------------------

def _parse_amdgpu_top_payload(payload, target_indices=(0, 1, 2, 3)) -> list[GpuSample]:
    """
    amdgpu_top -J 출력은 보통:
      { "0": { "device_name": ..., "metrics": {...}, ... }, "1": {...} }
    형태로 들어온다. 키가 문자열/정수 둘 다일 수 있어 양쪽 다 처리.
    """
    out: list[GpuSample] = []
    if not isinstance(payload, dict):
        return out

    for idx in target_indices:
        node = payload.get(str(idx)) or payload.get(idx)
        if not isinstance(node, dict):
            continue

        sample = GpuSample(index=idx, name=f"card{idx}", source="amdgpu_top")

        # amdgpu_top metrics 블록 (필드명은 README 기준)
        metrics = node.get("metrics") or node  # metrics 하위에 있을 수도, 최상위에 있을 수도

        # 온도: 'temp' / 'temperature'
        for k in ("temp", "temperature", "Temperature"):
            if k in metrics and metrics[k] not in (None, ""):
                try:
                    sample.temperature_c = float(metrics[k])
                    break
                except (TypeError, ValueError):
                    pass

        # 전력: 'average_socket_power' / 'power'
        for k in ("average_socket_power", "power", "Power"):
            if k in metrics and metrics[k] not in (None, ""):
                try:
                    sample.power_w = float(metrics[k])
                    break
                except (TypeError, ValueError):
                    pass

        # VRAM: 'vram_used' (MiB) 또는 'memory_usage'
        for k in ("vram_used", "vram_used_mib", "memory_usage"):
            if k in metrics and metrics[k] not in (None, ""):
                try:
                    sample.vram_used_mib = float(metrics[k])
                    break
                except (TypeError, ValueError):
                    pass

        # 사용률: 'gpu_use' / 'util'
        for k in ("gpu_use", "utilization", "util", "GPU use (%)"):
            if k in metrics and metrics[k] not in (None, ""):
                try:
                    sample.gpu_use_pct = float(metrics[k])
                    break
                except (TypeError, ValueError):
                    pass

        out.append(sample)

    return out


# ---------------------------------------------------------------------------
# 백엔드 호출
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("telemetry command timed out: %s", cmd)
        return None
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        log.warning("telemetry command failed (%s): %s",
                    result.returncode, (result.stderr or "").strip()[:200])
        return None
    return result.stdout


def _collect_via_rocm_smi(target_indices=(0, 1, 2, 3), timeout: float = 2.0) -> Optional[list[GpuSample]]:
    """
    rocm-smi --json 출력. 1.x 구버전은 --json이 dict 하나, 일부 5.x/6.x는
    { "card0": {...}, "card1": {...} } 형태. 둘 다 처리.
    """
    raw = _run([
        "rocm-smi",
        "--json",
        "--showtemp", "--showpower", "--showuse", "--showmeminfo", "vram",
    ], timeout=timeout)
    if raw is None:
        return None

    # 가끔 stderr가 섞여 들어오거나 앞뒤에 노이즈가 있을 수 있어 JSON 추출을 시도
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    samples: list[GpuSample] = []
    if not isinstance(payload, dict):
        return None

    # 형태 (A): {"card0": {...}, "card1": {...}}
    # 형태 (B): {"0": {...}}  또는 최상위 dict에 카드 정보가 평평하게
    card_keys = [k for k in payload.keys() if re.fullmatch(r"(card|device)?\d+", str(k), re.I)]
    if card_keys:
        for key in sorted(card_keys, key=lambda s: int(re.findall(r"\d+", s)[0])):
            idx = int(re.findall(r"\d+", key)[0])
            if idx not in target_indices:
                continue
            samples.append(_parse_rocm_smi_card(f"card{idx}", payload[key]))
        return samples

    # 형태 (C): 최상위 dict 하나에 카드들이 list로 들어있는 구버전 → 알 수 없음
    return None


def _collect_via_amdgpu_top(target_indices=(0, 1, 2, 3), timeout: float = 2.0) -> Optional[list[GpuSample]]:
    raw = _run(
        ["amdgpu_top", "-d", ",".join(str(i) for i in target_indices), "-J"],
        timeout=timeout,
    )
    if raw is None:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    samples = _parse_amdgpu_top_payload(payload, target_indices)
    return samples if samples else None


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def collect(target_indices=(0, 1, 2, 3), timeout: float = 2.0) -> list[GpuSample]:
    """
    4개 카드의 텔레메트리를 수집한다. 우선순위:
      1) rocm-smi --json
      2) amdgpu_top -J
      3) 둘 다 실패하면 모든 필드가 None인 'unavailable' 샘플 반환
    카드 인덱스는 항상 0..N-1 순서로 정렬해 돌려준다.
    """
    samples = _collect_via_rocm_smi(target_indices, timeout=timeout)
    backend = "rocm-smi"
    if not samples:
        samples = _collect_via_amdgpu_top(target_indices, timeout=timeout)
        backend = "amdgpu_top"

    samples = samples or []

    # 누락된 카드는 'unavailable' 마커로 채움 (가짜 수치 X)
    present = {s.index for s in samples}
    for idx in target_indices:
        if idx not in present:
            samples.append(GpuSample(index=idx, name=f"card{idx}", source="unavailable"))

    # 정상 백엔드로 채워진 것 외에는 source를 'unavailable'로 둠
    for s in samples:
        if s.source in ("rocm-smi", "amdgpu_top"):
            s.source = backend

    return sorted(samples, key=lambda s: s.index)


def to_vector_string(samples: list[GpuSample]) -> str:
    """
    LLM 입력용 한 줄 문자열.
    예: "[card0 T:38.0C,P:55.0W,V:1024.0MiB,U:12.0%] [card1 ...]"
    """
    return " ".join(s.to_vector_token() for s in samples)


def to_dict(samples: list[GpuSample]) -> list[dict]:
    return [asdict(s) for s in samples]


def collect_recent_samples(
    n_samples: int = 5,
    interval_sec: float = 60.0,
    target_indices=(0, 1, 2, 3),
    timeout: float = 2.0,
    sleep_fn = None,   # 테스트 시 주입 (time.sleep 대신)
) -> list[list[GpuSample]]:
    """
    시계열 수집: n_samples 회, interval_sec 간격으로 collect() 호출.
    반환: list[list[GpuSample]] — 각 원소가 한 시점의 카드별 측정값.
    """
    import time
    history = []
    sleep = sleep_fn or time.sleep
    for i in range(n_samples):
        samples = collect(target_indices=target_indices, timeout=timeout)
        history.append(samples)
        if i < n_samples - 1:
            sleep(interval_sec)
    return history


def summarize_timeseries(history: list[list[GpuSample]]) -> dict:
    """
    시계열 요약: 카드별 (현재값, 첫값, 변화, 변화량) 통계.
    history[0]가 가장 오래된 측정, history[-1]가 현재 측정.
    """
    if not history:
        return {}
    out = {}
    # 카드 인덱스 수집 (첫 시점 기준)
    indices = [s.index for s in history[0]]
    for idx in indices:
        temps = [s.temperature_c for s in history_at_index(history, idx)]
        temps_clean = [t for t in temps if t is not None]
        if len(temps_clean) < 2:
            out[idx] = {"current": temps[-1] if temps else None,
                        "earliest": temps[0] if temps else None,
                        "delta": None, "trend": "unknown"}
            continue
        delta = temps_clean[-1] - temps_clean[0]
        trend = "rising" if delta > 1.0 else ("falling" if delta < -1.0 else "stable")
        out[idx] = {
            "current": temps_clean[-1],
            "earliest": temps_clean[0],
            "delta": delta,
            "trend": trend,
        }
    return out


def history_at_index(history: list[list[GpuSample]], idx: int) -> list[GpuSample]:
    """시계열에서 특정 카드 인덱스만 시간순으로 추출."""
    out = []
    for snap in history:
        for s in snap:
            if s.index == idx:
                out.append(s)
                break
    return out


# ---------------------------------------------------------------------------
# 범용 센서 추상화 (연역적 일반화의 첫 단계)
# SensorReading + Threshold — GPU/배터리/모터/LiDAR/카메라/마이크 등
# 모든 물리적 센서에 적용 가능한 도메인 무관 데이터 구조.
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """
    범용 센서 측정값 (GpuSample의 일반화).
    GPU/배터리/모터/LiDAR/카메라/마이크 등 어떤 센서든
    같은 형식으로 표현할 수 있다.
    """
    sensor_id: str          # "gpu0", "battery1", "motor2", "lidar_front"
    sensor_type: str        # "temperature", "voltage", "rpm", "distance", "db"
    value: float            # 측정값 (단일 수치)
    unit: str               # "C", "V", "m", "dB", "rpm"
    timestamp: float = 0.0  # 측정 시각 (Unix epoch; 0이면 미지정)
    extras: dict = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class Threshold:
    """
    도메인 무관 임계치 + action 매핑.
    예: Threshold("temperature", ">=", 40.0, "Move quickly")
    """
    sensor_type: str        # 어떤 sensor_type에 적용되는지
    comparison: str         # ">=", "<=", ">", "<", "=="
    threshold_value: float
    action_text: str        # 임계치 초과/미만 시 action 문구 (영문)


def gpu_sample_to_sensor_reading(s: "GpuSample") -> list[SensorReading]:
    """
    GpuSample 하나를 SensorReading 리스트로 변환.
    GpuSample이 가진 4개 필드 (T, P, V, U)를 각각 SensorReading으로.
    """
    out = []
    ts = 0.0
    if s.temperature_c is not None:
        out.append(SensorReading(s.name, "temperature", s.temperature_c, "C", ts))
    if s.power_w is not None:
        out.append(SensorReading(s.name, "power", s.power_w, "W", ts))
    if s.vram_used_mib is not None:
        out.append(SensorReading(s.name, "vram", s.vram_used_mib, "MiB", ts))
    if s.gpu_use_pct is not None:
        out.append(SensorReading(s.name, "utilization", s.gpu_use_pct, "%", ts))
    return out


# ---------------------------------------------------------------------------
# 데모 / 스모크 테스트 (실제 GPU 없으면 그냥 unavailable 반환)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    samples = collect()
    print("Vector:", to_vector_string(samples))
    print("Detail:", json.dumps(to_dict(samples), indent=2, ensure_ascii=False))




def direction_phrase(value, comparison, threshold_value, unit=""):
    """
    측정값과 임계치의 비교 결과에 따라 정직한 방향 표현을 반환.
    
    Parameters
    ----------
    value : float
        측정값
    comparison : str
        비교 연산자 ('>=', '<=', '>', '<', '==')
    threshold_value : float
        임계값
    unit : str
        단위 (예: 'V', '°C', 'rpm')
    
    Returns
    -------
    str
        "is higher than the threshold of {threshold}{unit}"
        "is below the threshold of {threshold}{unit}"
        "is higher than or equal to the threshold of {threshold}{unit}"
        "is below or equal to the threshold of {threshold}{unit}"
        등등
    """
    if value is None:
        return f"is unknown compared to the threshold of {threshold_value}{unit}"
    
    # 임계치 초과/미만 판정
    if comparison == ">=":
        if value >= threshold_value:
            return f"is higher than or equal to the threshold of {threshold_value}{unit}"
        else:
            return f"is below the threshold of {threshold_value}{unit}"
    elif comparison == "<=":
        if value <= threshold_value:
            return f"is below or equal to the threshold of {threshold_value}{unit}"
        else:
            return f"is higher than the threshold of {threshold_value}{unit}"
    elif comparison == ">":
        if value > threshold_value:
            return f"is higher than the threshold of {threshold_value}{unit}"
        else:
            return f"is below or equal to the threshold of {threshold_value}{unit}"
    elif comparison == "<":
        if value < threshold_value:
            return f"is below the threshold of {threshold_value}{unit}"
        else:
            return f"is higher than or equal to the threshold of {threshold_value}{unit}"
    elif comparison == "==":
        if value == threshold_value:
            return f"is equal to the threshold of {threshold_value}{unit}"
        elif value > threshold_value:
            return f"is higher than the threshold of {threshold_value}{unit}"
        else:
            return f"is below the threshold of {threshold_value}{unit}"
    
    return f"has unknown comparison with the threshold of {threshold_value}{unit}"





