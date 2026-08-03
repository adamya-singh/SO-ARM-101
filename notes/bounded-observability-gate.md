# Bounded Observability Decision Gate

## Decision

The gate completed with terminal status **`closed_loop_not_resolved`**. Robot
and cube dynamics, causal pre-action contact/grasp state, and two-frame history
were not sufficient to turn the fixed 450-row oracle trajectory plus eight
isolated recovery labels into safe autonomous feedback control.

The observability-only branch is closed. The predefined next step is complete
oracle correction trajectories collected from policy-induced states
(DAgger-style), not a fourth feature schema or another tiny-policy
hyperparameter pass.

Immutable decision:
`artifacts/so_arm101_v2/oracle_distillation/observability_gates/
3cdb2c3d2c467a5b/report.json`.

## Controlled experiment

The following remained fixed for every candidate:

- 450 canonical nominal rows and eight recovery rows;
- equal recovery-row loss weight `1.0`;
- width 256, seed 101, full-batch Adam, and the fixed 30k maximum schedule;
- delta target, nominal-only normalization, offline thresholds, and command
  safety path;
- deterministic nominal MuJoCo evaluation with exactly three repeats.

Only the input schema changed, in the predefined order:

1. `phase_dynamics` (26 inputs);
2. `phase_dynamics_contact` (29 inputs);
3. `phase_dynamics_contact_history2` (57 inputs).

The causal annotation replay used no future state and left every oracle label
unchanged. All 450 nominal and eight recovery states replayed within fixed
field-specific numerical tolerances. Artifact:
`artifacts/so_arm101_v2/oracle_distillation/observability/
ec2d88b418e15186/manifest.json`.

## Results

| Candidate | Offline steps | Nominal MSE | Max ACT error | Nominal success | Repeated safety counts | Maximum height gain |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Existing 10-input recovery control | 24,691 | `9.9998e-7` | `0.005620` | `0/3` | 291 clipped, 0 limited, 13 unsafe | 5.173 mm |
| Phase + dynamics | 2,553 | `9.9602e-7` | `0.005822` | `0/3` | 385 clipped, 5 limited, 1 unsafe | 3.200 mm |
| Dynamics + contact | 2,266 | `9.8383e-7` | `0.005626` | `0/3` | 85 clipped, 0 limited, 0 unsafe | 0.061 mm |
| Dynamics + contact + history | 1,507 | `9.9969e-7` | `0.007374` | `0/3` | 74 clipped, 25 limited, 0 unsafe | 0.808 mm |

All policies passed the unchanged offline gate with zero training-command
safety violations. All closed-loop failures were exactly repeatable across
three deterministic runs. None acquired a strict grasp, completed pickup, or
entered successful placement.

Contact and history reduced clipping relative to the existing recovery
control, but this was not task improvement: the contact model never reached
meaningful contact, and the history model lifted less than 1 mm. The
dynamics-only model reached contact but regressed to more clipping and still
produced unsafe contact.

## Feature interpretation

The causal contact flags were identical between the nominal and recovery state
at all eight anchors. The preceding pre-action history was also identical,
because each perturbation was issued from the same preceding state. Therefore
contact and history did not add a discriminating signal between the eight
isolated anchor pairs. Dynamics did separate the perturbed states—most strongly
near first contact, seating, transport, and placement—but that separation did
not yield safe autonomous control.

Feature telemetry:
`artifacts/so_arm101_v2/oracle_distillation/observability_analyses/
3a6f3c32167342fc/report.json`.

This rejects the narrow hypothesis that omitted instantaneous dynamics,
contact state, or two-frame history alone explains the failure. It does not
show that observability is irrelevant generally. The experiment still contains
only eight isolated corrections, so the dominant unresolved limitation is
continuous coverage of the states the learned policy actually visits.

## Next controlled tranche

1. Roll out the current policy under the unchanged nominal task and capture
   complete oracle corrections from its induced states, including approach,
   contact, closure, and post-contact divergence—not isolated single rows.
2. Store the correction trajectories as a new immutable dataset version while
   retaining the 450-row nominal source and full provenance.
3. Retrain one fixed supervised control with unchanged architecture and
   optimization before considering a stronger model.
4. Require the same offline safety checks and nominal MuJoCo `3/3` gate.
5. Stop if the controlled data addition does not improve closed-loop behavior;
   do not compensate with simultaneous feature, capacity, or optimizer changes.

Width 512, vision, ACT, RL, broader evaluation, and physical deployment remain
unauthorized until a supervised candidate passes nominal closed-loop control.
