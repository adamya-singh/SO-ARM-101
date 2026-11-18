# Pixel-reliance ablation and architecture comparison (2026-11-18)

Question: do our latest policies actually use the camera? (Prompted by the zecice SO-101 ping-pong balancer,
whose README says an end-to-end CNN "barely saw the real ball"; it gives no method for that claim.)

Method: `tools/eval_policy.py --black-image` (new; the VLA path had no switch, the CNN path did) replaces every
frame the policy sees with zeros. Same 30 held-out SO-101 placements
(`bench_pick_replace_v1_seed8_n10_appearance_placement_0a75289f941d_so101`), same tool, real images vs black.

| Policy | Scene | Real images | Black images |
|---|---|---|---|
| ACT high_lift_20261112, execute 30 | so101_servo_lift100 | 21/30 (existing eval) | 0/30, all safety_invalidation, never descended, lateral median 113 mm |
| CNN perturbed_so101 cnn_s202 (v2 encoder) | so101 | 21/30, lateral median 5.9 mm, 24/30 within tolerance | 0/30, all pickup_incomplete, lateral median 112 mm |

Reading: both policies get the cube position from pixels; without them each falls back to a mean reach ~11 cm
off. This says nothing about reading REAL frames correctly (that is the real-frame gate and the live runs).

Evidence: `experiments/high_lift_20261112/evals/lift_act_ft_exec30__servo__black_image/`,
`experiments/perturbed_so101_20261108/evals/cnn_s202__so101__{normal,black}/`.

Architecture comparison page (3D, per-layer reasoning): `notes/policy-anatomy.html`, published as
https://claude.ai/artifact/L6RQWTtkb8NuUyWF5EQgs8. ACT 51.6M parameters (34.2M at run time; CVAE encoder
17.3M training only); CNN v2 2.71M; the balancer's MLP sizes are not stated in its README.

Also committed: `tools/claude_arm.py` (2026-10-04, Claude driving the arm step by step). Its jaw-height
guard trusts the joint map, which put the jaw 33-38 mm above the table while it was touching it; do not rely
on that guard until the joint zeros are measured (bubble level + touch-off).
