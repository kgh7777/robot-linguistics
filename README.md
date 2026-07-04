# Sensor Grounding Essay Framework

A deterministic sensor-grounding essay generation framework for robotic communication.

## What this is

A library that converts **live sensor measurements** into **honest, natural-language essays** without using LLM hallucination. The pipeline is:

[Physical Sensor] → [Sensor Reading] → [Essay Composition] → [English/Korean Text]
(e.g. Radeon GPU 48.0°C)        (deterministic)              (zero hallucination)


The framework produces essays that:
- **Cite actual sensor values** (not hallucinated numbers)
- **Distinguish the four cases of threshold comparison** correctly (≥/≤, with/without action)
- **Work across multiple sensor domains** (GPU, battery, motor, LiDAR, camera, mic)
- **Are deterministic** — the same input produces the same output, every time

## Why this matters

Traditional LLM-based systems generate text by sampling from a learned distribution.
This means a temperature reading of 48.0°C may be rephrased as "approximately 50 degrees"
or "around 47 to 50" — **a hallucination of 2-3°C**.

This framework takes a different approach: it **bypasses the LLM's distribution entirely**
and uses deterministic functions that take the actual sensor value as input and emit
text that is **provably faithful to the measurement**.

The LLM (Qwen2.5 32B) still runs and produces the language, but its role is limited to
**vocabulary choice within a fixed grammar**, not free-form text generation.

## What's been verified (live)

| # | Verification | Status |
|---|---|---|
| 1 | 18 live runs of Radeon GPU 4-way essay (RDNA3 + RDNA2) | ✓ |
| 2 | 4-way × time-series essay (essay-timeseries mode) | ✓ |
| 3 | 4 cases of threshold comparison (≥/≤ × branch 1/2) — Battery essay | ✓ |
| 4 | 4 cases × 3 domains = 12 combinations (GPU / Battery / Motor) | ✓ |
| 5 | Multi-domain time-series essay (3 time steps × 3 domains) | ✓ |
| 6 | Grounding dataset (9 essays with metadata) | ✓ |

## The 4 cases of threshold comparison (all verified correct)

| Case | value vs threshold | comparison | branch | direction phrase |
|---|---|---|---|---|
| 1 | 12.4V > 11.0V (above) | `<=` (else) | "is higher than the threshold of 11.0V" |
| 2 | 10.5V <= 11.0V (below) | `<=` (if) | "is below or equal to the threshold of 11.0V" + "Return to base" |
| 3 | 12.4V >= 11.0V (above) | `>=` (if) | "is higher than or equal to the threshold of 11.0V" + "Alert" |
| 4 | 10.5V < 11.0V (below) | `>=` (else) | "is below the threshold of 11.0V" |

## Quick start

```python
import sys
sys.path.insert(0, '/home/kgh')
import importlib

import gpu_telemetry as gt
gp = importlib.import_module('grounding_pipeline01')

# Single-domain, single-timestep
readings = [gt.SensorReading('battery1', 'voltage', 12.4, 'V')]
thresholds = [gt.Threshold('voltage', '<=', 11.0, 'Return to base')]
print(gp.compose_essay_universal(readings, thresholds))

# Multi-domain, time-series
history = {
    'gpu': [...],
    'battery': [...],
    'motor': [...],
}
thresholds = {
    'gpu': [gt.Threshold('temperature', '>=', 40.0, 'Move quickly')],
    'battery': [gt.Threshold('temperature', '<=', 11.0, 'Return to base')],
    'motor': [gt.Threshold('temperature', '>=', 80.0, 'Stop motor')],
}
print(gp.compose_essay_timeseries_multidomain(history, thresholds))

Files
gpu_telemetry.py — Sensor telemetry + SensorReading, Threshold data classes + collect_recent_samples() for time-series
grounding_pipeline01.py — The essay composition functions:

compose_paragraph_universal() — 5-sentence paragraph (domain-agnostic, with direction-aware threshold comparison)
compose_essay_universal() — 3-paragraph essay (Introduction + Body + Conclusion)
compose_essay_timeseries_multidomain() — Time-series × multi-domain essay


build_grounding_dataset.py — Builds the grounding dataset (9 essays, JSON)
grounding_dataset.json — The grounding dataset (output)
ConceptsCausal grounding, not philosophical meaningThe essays produced by this framework do not "mean" anything in the human-philosophical
sense. They are causally grounded: a sensor reading of 48.0°C is faithfully rendered
as "The temperature is 48.0°C", and an action of "Move quickly" is reliably followed
by motor actuation. This is causal grounding — sufficient for robotic communication,
distinct from human philosophical meaning.Bender & Koller (2020) argue that octopus-like systems cannot have meaning without
physical experience. We accept this for humans, but argue that for robots composed
of sensors + LLM + actuators, causal grounding is sufficient and necessary —
not philosophical meaning.Two languages, not oneWe propose two languages for robotic communication:
Robot Language (RL): for robot-to-robot communication. Tokens are causally
grounded sensor readings, not human-understandable. Example: "GPU.Temperature=48C;
Battery.Voltage=10.5V; Motor.RPM=3000".
Robot-Human Common Language (RHCL): for robot-to-human communication. Same
sensor readings, rendered as natural-language essays. Example: "The temperature
is 48.0°C. Move quickly. The system is stable, however, so the response remains
under control."
The framework produces RHCL today. RL is a future extension.Robot LinguisticsA new field, Robot Linguistics, is proposed:
Nouns = physical events (temperature, voltage, distance, time)
Verbs = physical actions (Move, Stop, Reduce, Increase)
Adjectives = physical states (High, Low, Stable, Critical)
Adverbs = physical quantities (Quickly, Slowly, Steadily)
Tense/Aspect/Mood = physical time
Speech acts = physical behaviors (declarative, imperative, conditional, concessive)
Limitations
English only in the current version. Korean essays are a 1-sentence summary only.
No ROS 2 integration — the framework is sensor-mock-friendly but not connected to
live robot sensors yet.
No LLM comparison evaluation — we have not yet benchmarked Qwen vs Llama vs GPT-4
on this grounding task.
Small dataset (9 essays) — not yet statistically meaningful.
No theory of action-to-actuator mapping — actions ("Move quickly") are produced
but not yet connected to motor control signals.
Future work
ROS 2 integration: connect essay actions to motor control signals
LLM comparison: Qwen vs Llama vs GPT-4 on grounding accuracy
Larger dataset: 100-1000 essays across more domains and time-series patterns
Korean essays: full 3-paragraph KO essays
Robot Language (RL): tokens for robot-to-robot communication
Robot Linguistics: formalize the field with the present framework as the first case
LicenseMITAcknowledgmentsBuilt on Radeon GPU 4-way cluster (RDNA3 + RDNA2), Qwen2.5 32B, and the user's persistence
in pursuing the question of "what does it mean for a robot to speak honestly".




We thank Chatly (chatly.ai) and Google for AI-assisted code generation and debugging during this research
