"""
robot_linguistics_cycle.py

Qwen → Llama → Qwen 3자 사이클.
로봇-LLM-로봇 의사소통의 첫 라이브 사이클.

1) Qwen (RDNA3, port 2242)이 rocm-smi로 GPU 측정값을 읽고 영어 essay 발화
2) Llama (RDNA2, port 11434)이 Qwen essay를 읽고 추가 명령 생성
3) Qwen이 Llama 명령을 읽고 응답
"""

import sys
sys.path.insert(0, '/home/kgh')

import json
import time
import urllib.request

import gpu_telemetry as gt
import importlib
gp = importlib.import_module('grounding_pipeline01')


# 두 LLM 서버의 endpoint
QWEN_URL = "http://localhost:2242/v1/chat/completions"
LLAMA_URL = "http://192.168.1.171:11434/v1/chat/completions"


def call_llm(url: str, prompt: str, model: str, max_tokens: int = 256) -> str:
    """LLM에 HTTP 요청."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # 결정적
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        return result['choices'][0]['message']['content']


def main():
    print("=" * 60)
    print("Qwen → Llama → Qwen 3자 사이클")
    print("=" * 60)

    # 1단계: GPU 측정값 → Qwen essay
    print("\n[1단계] Qwen (RDNA3 2242): rocm-smi 측정값 → 영어 essay")
    try:
        samples = gt.collect(target_indices=(0,), timeout=2.0)
    except Exception as e:
        print(f"  rocm-smi 실패, mock 사용: {e}")
        # mock GPU 측정값
        from dataclasses import dataclass
        @dataclass
        class MockSample:
            temperature_c: float = 48.0
            power_w: float = 25.0
            vram_mib: float = 1024.0
            utilization_pct: float = 1.0
        samples = [MockSample()]
    
    # GpuSample 형식으로 변환
    gpu_samples = []
    for s in samples:
        gpu_samples.append(gt.GpuSample(0, 'card0', 'rocm-smi',
                                         temperature_c=s.temperature_c,
                                         power_w=s.power_w,
                                         vram_mib=s.vram_mib,
                                         utilization_pct=s.utilization_pct))
    
    # essay 생성
    essay = gp.compose_essay_universal(gpu_samples, [gt.Threshold('temperature', '>=', 40.0, 'Move quickly')])
    print(f"  Essay:\n{essay}\n")

    # 2단계: Qwen essay → Llama 명령
    print("\n[2단계] Llama (RDNA2 11434): Qwen essay 읽고 → 모터 명령 생성")
    prompt_for_llama = f"""You are a motor control agent. The following English essay was produced by another AI agent based on real-time GPU sensor readings:

{essay}

Read the essay carefully. If the essay says the GPU requires action, generate a motor command in this exact format:

MOTOR_COMMAND: <action>; SPEED_CHANGE: <percentage>; REASON: <one-sentence reason based on the essay>

If the essay says the system is stable, generate:
MOTOR_COMMAND: NONE; REASON: <one-sentence reason based on the essay>

Be brief and specific. Do not invent facts; base your command on the essay's actual content."""

    try:
        llama_response = call_llm(LLAMA_URL, prompt_for_llama, model="llama3.3:70b-instruct-q4_K_M", max_tokens=200)
        print(f"  Llama 응답:\n{llama_response}\n")
    except Exception as e:
        print(f"  Llama 호출 실패: {e}")
        llama_response = "(no Llama response)"

    # 3단계: Llama 명령 → Qwen 응답
    print("\n[3단계] Qwen (RDNA3 2242): Llama 명령 읽고 → 응답 생성")
    prompt_for_qwen = f"""You are a robot motor control agent. You previously generated this English essay based on GPU sensor readings:

{essay}

Now you received the following motor command from another agent:

{llama_response}

Acknowledge the command in this format:

ACKNOWLEDGED: <yes/no>; ACTION_TAKEN: <what you would do>; NEXT_STEP: <one-sentence next step>

Be brief. Do not invent facts; base your response on the essay and the command."""

    try:
        qwen_response = call_llm(QWEN_URL, prompt_for_qwen, model="qwen2.5:32b-instruct-q8_0", max_tokens=200)
        print(f"  Qwen 응답:\n{qwen_response}\n")
    except Exception as e:
        print(f"  Qwen 호출 실패: {e}")
        qwen_response = "(no Qwen response)"

    # 결과 저장
    cycle_result = {
        'qwen_essay': essay,
        'llama_command': llama_response,
        'qwen_response': qwen_response,
        'timestamp': time.time(),
    }
    
    output_path = '/home/kgh/robot_linguistics_cycle.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cycle_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[완료] 결과가 {output_path}에 저장됨")


if __name__ == '__main__':
    main()
