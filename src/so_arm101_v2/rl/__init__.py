"""Reinforcement learning on the v2 bench simulator (retry tranche, 2026-10-04).

The task contract stays reward-free (tests/test_task_contract.py): rewards are defined here, from the
contract's own measurements and events, and nowhere else. Modules:

- ``reward``     sparse success / safety terminal reward + bounded potential shaping on contact milestones
- ``env``        ``ChunkEnv``: one adapter stepped a chunk of commands at a time, frame rendered at boundaries
- ``vec_env``    ``SubprocChunkVecEnv``: spawn workers (no torch), async step / wait
- ``scenarios``  training and monitoring suites disjoint from the held-out and capture seeds; fixed order
- ``ppo``        per-episode GAE, clipped surrogate, KL early stop
- ``critic``     privileged-state value function (asymmetric actor-critic)
- ``reinflow``   ReinFlow sampler + noise head over any flow model (pi0.5, SmolVLA)
- ``act_ppo``    Gaussian chunk policy over ACT
- ``collector``  batched-inference rollout collection over the vector env
"""
