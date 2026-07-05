#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# 1단계 규격: 범용 센서 데이터 구조
class SensorReading:
    def __init__(self, sensor_id, sensor_type, value, unit):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.value = value
        self.unit = unit

DEFAULT_THRESHOLDS = {"temp_high_c": 70.0}

# 4단계 규격: 소문자 후처리 방어 가드가 탑재된 액추에이터 매핑 테이블
ACTION_MAP = {
    "move quickly": {"type": "motor", "speed_pct": 80, "direction": "forward"},
    "reduce load": {"type": "motor", "speed_pct": 30, "direction": "forward"},
    "stop motor": {"type": "motor", "speed_pct": 0, "direction": "stop"},
    "return to base": {"type": "navigation", "goal": "home"},
    "alert": {"type": "alert", "level": "warning"},
}

def action_to_actuator(actions):
    """자연어 발화 텍스트 리스트를 하드웨어 제어 지령 구조로 결정론적 변환"""
    actuator_commands = []
    for action in actions:
        if not action:
            continue
        clean_action = action.strip().rstrip(".").lower()
        if clean_action in ACTION_MAP:
            actuator_commands.append(ACTION_MAP[clean_action])
        else:
            actuator_commands.append({"type": "unknown", "action": action})
    return actuator_commands

def compose_paragraph(sample, threshold_c=70.0, action=""):
    """3단계: 5문장 학술 단락 자연어 격자화 생성기"""
    clean_act = action.strip().rstrip(".") if action else ""
    topic = f"GPU telemetry is being monitored because the system has detected an operational issue." if clean_act else "GPU telemetry is being monitored to maintain stable device operation."
    detail = f"The temperature is {sample.value:.1f}{sample.unit}."
    evidence = f"The temperature is higher than the threshold ({sample.value:.1f}°C >= {threshold_c:.1f}°C)." if clean_act else f"The temperature is lower than the threshold ({sample.value:.1f}°C < {threshold_c:.1f}°C)."
    action_sent = f"{clean_act.upper() + clean_act[1:]}." if clean_act else ""
    conclusion = "The system is under stress due to high temperature, however, so the response must remain within safety limits." if clean_act else "The system is stable, however, so the response remains under control."
    return " ".join(p for p in [topic, detail, evidence, action_sent, conclusion] if p)

class RadeonCausalBridgeNode(Node):
    def __init__(self):
        super().__init__('radeon_causal_bridge_node')
        # 토픽 퍼블리셔 및 모니터링용 서브스크라이버 동시 가동
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.motor_callback, 10)
        self.timer = self.create_timer(1.0, self.control_loop)
        self.get_logger().info("🤖 [로봇 언어학 마스터 랩] 마스터 도킹 제어 노드가 가동되었습니다.")

    def mock_rocm_smi(self):
        # 10초 주기로 45.2도와 74.5도를 오가며 실시간 가상 피드백 생성
        cycle = int(time.time()) % 20
        simulated_temp = 74.5 if cycle >= 10 else 45.2
        return SensorReading("gpu0", "temperature", simulated_temp, "°C")

    def control_loop(self):
        # [1단계 & 2단계] 사건 감지 및 결정론적 액션 트리거 도출
        reading = self.mock_rocm_smi()
        th_c = DEFAULT_THRESHOLDS["temp_high_c"]
        action = "Stop motor" if reading.value >= th_c else ""
        
        # [3단계] 고등 자연어 프로토콜 형성
        en_paragraph = compose_paragraph(reading, threshold_c=th_c, action=action)
        
        print(f"\n📡 [3단계 자연어 발화 생성]")
        print(f"  🇺🇸 EN: {en_paragraph}")

        # [4단계 마스터 도킹] 언어 프로토콜에서 액션을 추출하여 액추에이터 명령어로 번역
        extracted_actions = []
        if "stop motor" in en_paragraph.lower():
            extracted_actions.append("Stop motor")
        
        # 만약 발화문 내에 명시적인 액션 지시가 없다면 정상 주행(Move quickly 혹은 기본 주행)으로 변환
        if not extracted_actions and reading.value < th_c:
            # 정상 가동 시에는 기본 과부하 완화 주행 상태인 'reduce load'를 기본 기호로 주입한다고 가정
            extracted_actions.append("reduce load")

        commands = action_to_actuator(extracted_actions)
        print(f"⚙️ [액추에이터 명령어 해석 결과]: {commands}")

        # 물리 구동 지령으로 최종 사상 (Twist 휠 제어 값 연산)
        twist_cmd = Twist()
        if commands and commands[0]["type"] == "motor":
            # speed_pct(30% 또는 0%)를 바탕으로 실시간 물리 속도 지령 환산 (Max 속도를 1.0 m/s로 기준 잡음)
            target_speed = float(commands[0]["speed_pct"]) / 100.0
            twist_cmd.linear.x = target_speed
            
        self.cmd_vel_pub.publish(twist_cmd)

    def motor_callback(self, msg):
        # 전선에 인입된 최종 물리 지령 수치를 실시간 모니터링 로그로 전사
        self.get_logger().info(f"🏎️ [4단계 하드웨어 피드백] 최종 가상 바퀴 구동 출력 속도: {msg.linear.x:.2f} m/s")

def main(args=None):
    rclpy.init(args=args)
    node = RadeonCausalBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
