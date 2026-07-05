# fix_compose_paragraph.py
import re

with open('/home/kgh/grounding_pipeline01.py', 'r') as f:
    code = f.read()

new_body = '''def compose_paragraph_universal(readings, thresholds) -> str:
    """5문장 단락 (도메인 무관, 비교 방향 보강 포함)."""
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
                if check_threshold(value, th):
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
                if not check_threshold(value, th):
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

    if all_actions:
        parts.append("The system has detected the condition and will respond accordingly.")
    else:
        parts.append("The system is stable and continues to monitor.")

    return " ".join(parts)
'''

# 첫 번째 def compose_paragraph_universal 함수를 new_body로 교체
pattern = r'def compose_paragraph_universal\(readings, thresholds\) -> str:.*?(?=\ndef |\Z)'
new_code = re.sub(pattern, new_body.strip() + '\n\n\n', code, count=1, flags=re.DOTALL)

with open('/home/kgh/grounding_pipeline01.py', 'w') as f:
    f.write(new_code)

print("OK: compose_paragraph_universal 교체 완료")
