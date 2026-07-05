"""
action_to_actuator.py

action 문구 → actuator 제어 신호 변환.
지금까지 검증된 essay의 action 발화를 실제 모터 제어 신호로 변환.

지원하는 action:
  - "Move quickly"       → 모터 속도 +50%
  - "Stop motor"          → 모터 정지 (0%)
  - "Reduce load"         → 모터 속도 -30%
  - "Return to base"      → 역추적 경로 활성화
  - "Free memory"         → 메모리 정리 명령
  - "Alert"               → 경보 활성화
  - "Heat motor"          → 히터 활성화
"""

import time


# Action 문구 → actuator 제어 명령 매핑
ACTION_TO_ACTUATOR = {
    "Move quickly": {
        "actuator": "motor",
        "command": "speed_up",
        "pwm_delta": +50,  # 50% 증가
    },
    "Stop motor": {
        "actuator": "motor",
        "command": "stop",
        "pwm_value": 0,  # 정지
    },
    "Reduce load": {
        "actuator": "motor",
        "command": "speed_down",
        "pwm_delta": -30,  # 30% 감소
    },
    "Return to base": {
        "actuator": "navigation",
        "command": "return_to_base",
        "waypoint": "home",
    },
    "Free memory": {
        "actuator": "memory",
        "command": "gc_collect",
    },
    "Alert": {
        "actuator": "buzzer",
        "command": "alert",
        "duration_sec": 5,
    },
    "Heat motor": {
        "actuator": "heater",
        "command": "activate",
        "duration_sec": 60,
    },
}


def action_to_actuator_command(action_text: str) -> dict:
    """
    action 문구 → actuator 제어 명령으로 변환.
    
    Args:
        action_text: essay에서 발화된 action 문구 (예: "Move quickly")
    
    Returns:
        dict: {
            'action': str,
            'actuator': str,
            'command': str,
            'pwm_delta': int (optional),
            'pwm_value': int (optional),
            ...
        }
        또는
        dict: {'action': action_text, 'actuator': 'unknown', 'command': 'noop'}
        (모르는 action이면 noop 반환)
    """
    if action_text in ACTION_TO_ACTUATOR:
        cmd = dict(ACTION_TO_ACTUATOR[action_text])
        cmd["action"] = action_text
        cmd["timestamp"] = time.time()
        return cmd
    else:
        return {
            "action": action_text,
            "actuator": "unknown",
            "command": "noop",
            "timestamp": time.time(),
        }


# ROS 2 호환 인터페이스 (옵션)
def action_to_ros2_cmd_vel(action_text: str) -> dict:
    """
    ROS 2 cmd_vel (geometry_msgs/Twist) 형식으로 변환.
    모터 제어 표준 메시지.
    """
    if action_text == "Move quickly":
        return {
            "linear": {"x": 0.5},  # m/s
            "angular": {"z": 0.0},
        }
    elif action_text == "Stop motor":
        return {
            "linear": {"x": 0.0},
            "angular": {"z": 0.0},
        }
    elif action_text == "Reduce load":
        return {
            "linear": {"x": 0.2},
            "angular": {"z": 0.0},
        }
    else:
        return {
            "linear": {"x": 0.0},
            "angular": {"z": 0.0},
        }
