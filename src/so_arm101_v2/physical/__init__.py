"""Physical execution of learned policies: one control loop, a simulator backend and a LeRobot backend.

The loop in :mod:`runner` mirrors the simulator's closed-loop evaluation step
for step (read pose, observe only when the policy needs a frame, gate with the
shared bench rule, send, wait one control period). :mod:`sim_backend` drives a
``MujocoTaskAdapter`` through that loop and cross-checks the gate against the
adapter on every step, which is how the physical code path is proven
equivalent to the evaluation that produced the policy's score.
:mod:`lerobot_backend` is the real arm and wrist camera.
"""
