# PIP — emotion-aware desktop robot (Reachy-Mini-style)

numpy-only brain + sim (Python 3.9+, any OS). CAD needs `pip install cadquery`.

    python scripts/train_sim.py 40     # pretrain in the virtual home, then idle-learning adaptation test
    python scripts/test_monitor.py     # routine learning + anomaly detection vs injected ground truth
    python scripts/demo_all.py         # 14-day run: emotions, alerts, voice commands, background learner
    python cad/pip_cad.py              # parametric CAD -> cad/out/*.step (see cad/ONSHAPE.md)

| module | role |
|---|---|
| `pip/params.json` | single source of truth for geometry (CAD + kinematics + sim) |
| `pip/kinematics.py` | Stewart-platform IK, workspace projection, servo lag/limits, body yaw, antennas |
| `pip/sim.py` | virtual home: adult/child/elder/cat, doors, meals, noise, visitors, injected anomalies, simulated-human feedback |
| `pip/brain.py` | emotion net (valence/arousal/6 moods, recurrent via fed-back state), expression policy+value (AWR), proactivity-preference net |
| `pip/monitor.py` | learns routines; alerts: missed meal, door left open, inactivity, unusual-hour activity |
| `pip/assistant.py` | neural intent classifier + smart-home registry (swap backend for Home Assistant/MQTT), timers |
| `pip/log.py`, `pip/learner.py` | private replay log; idle-gated, duty-cycled shadow fine-tuning with regression guard + atomic swap |

## Measured (this sandbox)
- Sim: ~30k steps/s (5 days in 0.16 s). Neural tick ~0.1 ms.
- Expression reward: -0.39 untrained -> 0.72 after 40 sim days; emotion valence error 0.024.
- Monitor (120 days, 15%/day anomaly rate, 10 clean learning days): 79/79 injected anomalies detected, 3 false alerts.
- User-preference change: reward fell 0.72 -> 0.64; six idle-learning nights recovered it to ~0.70.

## Honest limits
- The sim's "right emotion/expression" teacher and the human-feedback model are hand-written *training stand-ins*. On the real robot, replace them with real signals: explicit reactions (pet/tap/thanks/dismiss), user corrections, and smile/voice change after PIP acts. Expect a retune.
- Perception (face/voice emotion, speech-to-text, TTS, pet/person detection) is not included; `featurize()` expects its outputs (valence/arousal per modality). Use small on-device models.
- Routines are not weekday/weekend-aware yet; alert thresholds need tuning on real homes. Alerts to family members: wire `Alert` to push/SMS.
- Child/elder monitoring is sensitive: keep data local, disclose it to household members, and treat PIP as a convenience, not a safety device.
- CAD is Reachy-Mini-style, not Pollen's exact geometry, and was geometry-checked via IK only (not printed/FEA'd).
