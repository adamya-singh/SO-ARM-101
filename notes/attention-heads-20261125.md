# What each attention head of the ACT has learned (2026-11-25)

Question: the high-lift ACT (`experiments/high_lift_20261112/act_ft`, the policy of the live runs) has 48 attention
heads. What does each one read, and which of them does the plan depend on?

Answer: no head has a job of its own. Removing any one head moves the 90-step plan by at most 2.0% of its variance.
What the attention as a whole does is mix the joint state into the image tokens (encoder layers 1 and 2) and read the
result out (decoder cross-attention). It does not point at the cube: hiding the napkin patches from every attention
layer changes nothing. The style token is unused and the decoder's 8 self-attention heads cannot do anything.

## Method

`tools/attention_heads.py` (new), test `tests/test_attention_heads.py`. No training and no rollouts.

- Data: the run's held-out capture (9 placements, 480 frames each), every 6th frame: 720 frames. Each frame is a state
  the teacher reached, so this is open loop.
- The tool swaps every `nn.MultiheadAttention` of the encoder and decoder for a written-out copy (`HookedMHA`) whose
  heads can be read, replaced and blinded. The copy matches the module to 1.3e-6 on the plan, and the intact model
  scores 2.88e-4 on chunk start 90, the value the training run logged.
- Tokens: the encoder sees 66 tokens: style (the CVAE latent, zero at run time), state (6 joints and progress) and 64
  image patches of 32 px. Heads are named as in the files: `enc0` to `enc3` are encoder layers 0 to 3, `dec_self` and
  `dec_cross` are the two attentions of the one decoder layer, heads 0 to 7.
- Reading: per head, the share of attention that goes to the state token, the style token and the image; how many
  tokens it spreads over (`effective_keys`); how alike its pattern is from frame to frame (`consistency`, 1 = always
  the same); and how much it favours the napkin and the cube over the rest of the image (`lift`, 1 = no preference).
  The napkin is found as the largest bright region of the frame, the cube as the dark area inside its hull.
- Removing a head (mean ablation): its output is replaced by its mean over the 720 frames, so it can no longer pass
  on anything about the current frame. `effect` is how far the plan moves, as a share of the plan's variance
  (mean over the 6 joints). `ratio` is the error against the teacher divided by the intact model's.
- Hiding tokens: attention is stopped from reading a group of tokens and the weights renormalise.

## Results

Removing heads (`heads.json`):

| What is removed | Effect on the plan | Error against the teacher |
|---|---|---|
| nothing (intact model) | 0 | x1.00 (3.60e-3 over all frames) |
| the strongest single head, `enc1` head 5 | 0.0200 | x1.05 |
| any other single head | 0.0142 or less | x0.93 to x1.03 |
| all 8 heads of `enc0` | 0.0197 | x0.94 |
| all 8 heads of `enc1` | 0.1045 | x1.96 |
| all 8 heads of `enc2` | 0.0984 | x2.86 |
| all 8 heads of `enc3` | 0.0888 | x1.46 |
| all 8 heads of `dec_self` | 0.0000 | x1.00 |
| all 8 heads of `dec_cross` | 1.0173 | x13.38 |

Hiding tokens (`block.json`):

| What attention cannot read | Error against the teacher |
|---|---|
| nothing (intact model) | x1.00 |
| the state token, anywhere | x11.70 |
| the state token, in the encoder | x11.56 |
| the state token, in encoder layer 0 / 1 / 2 / 3 only | x1.00 / x3.06 / x5.17 / x1.09 |
| the state token, in the decoder | x0.98 |
| the style token, anywhere | x0.99 |
| style and state, in the decoder | x1.28 (first 30 plan steps x1.85) |
| any image token, in the decoder | x12.78 |
| the napkin patches (7.5 a frame), anywhere | x1.00 |
| as many other patches (control), anywhere | x1.01 |

What the heads read:

| Heads | What they read | Effect of removing one |
|---|---|---|
| `enc0` 0, 1, 2, 4, 5 | Fixed background regions; they avoid the napkin (lift 0.0 to 0.3). Head 4 puts 0.26 on the state. | 0.0001 to 0.0009 |
| `enc0` 3, 7 | A spot at the jaws and cube (cube lift 2.0 and 0.9). | 0.0029, 0.0024 |
| `enc0` 6 | 2.9 tokens: 0.73 on the state, the rest on the cube (lift 11.3). | 0.0023 |
| `enc1` 5 | 0.63 on the state, the rest favours the napkin (lift 2.5), 5.8 tokens. | 0.0200 |
| `enc1` 7 | 0.68 on the state. | 0.0044 |
| `enc1` 0, 3, 4, 6 | State (0.27 to 0.49) and image. | 0.0007 to 0.0031 |
| `enc1` 1, 2 | Image only. | 0.0022, 0.0032 |
| `enc2` 3 | 0.94 on the state, the same pattern every frame (consistency 0.89). | 0.0008 |
| `enc2` 0, 5, 6 | About half on the state (0.46 to 0.58), the rest favours the napkin (lift 1.4 to 2.3). | 0.0016 to 0.0037 |
| `enc2` 1, 4 | The least fixed patterns (consistency 0.38, 0.39); napkin lift 2.1 and 2.2. | 0.0034, 0.0018 |
| `enc2` 2, 7 | Broad image mixing (17.7 and 25.4 tokens). | 0.0023, 0.0010 |
| `enc3` 0 to 5 | Image only (state 0.08 or less), spread over 31 to 37 tokens. | 0.0005 to 0.0018 |
| `enc3` 6, 7 | Still read the state (0.27, 0.43). | 0.0014, 0.0059 |
| `dec_self` 0 to 7 | Nothing: see below. | 0 (2e-14) |
| `dec_cross` 0, 1, 2, 4, 5, 6 | Plan steps 0 to 29 read state (0.12 to 0.27) and style (0.12 to 0.26); later steps read the image. | 0.0053 to 0.0131 |
| `dec_cross` 3 | The image at every step (0.97). | 0.0142 |
| `dec_cross` 7 | The reverse: steps 60 to 89 read state (0.18) and style (0.12). | 0.0038 |

Every `dec_cross` head spreads over 36 to 41 of the 66 tokens and has no preference for the napkin (lift 0.8 to 1.2).

Figures: `overlay_<module>.png` (8 heads on 6 frames), `dec_cross_bystep.png` (what each decoder head reads at each
plan step), `attn_<module>.png` (one strip of 8 tiles on episode 0 step 150, used in the course presentation).

## Reading

1. The work is spread over heads. The single-head effects of a layer add up to far less than removing the layer
   (`enc2`: 0.018 against 0.098), so heads cover for each other. Do not read a job into one head.
2. The encoder's measurable job is the joint state. Without it the error is 11.6 times larger, and layers 1 and 2 are
   where it is read. Layer 0 can be removed whole without raising the error (x0.94).
3. Heads look at the napkin and cube (layer 2 especially) but the plan does not need them to. Hiding those patches
   from all attention leaves the error at x1.00. Untested explanation: the last ResNet-18 stage sees the whole
   256 px frame from every cell, so the cube's position is already in all 64 tokens.
4. The style token carries nothing of its own (x0.99 when hidden everywhere). The decoder reads it as a second copy
   of the state: hiding either does nothing, hiding both costs x1.28.
5. `dec_self` is dead by construction. The decoder input is zeros and there is one decoder layer, so the values it
   attends over are all the same vector. Its output is the same for all 90 queries to 7e-9.

## Limits

- Open loop on teacher-driven frames: error against the teacher's plan, not task success. No rollout was run with a
  head removed.
- One checkpoint, one seed, simulation only, 720 frames from 9 placements.
- Where a head looks is not what the plan needs (reading 3 is the example). The token-hiding and removal numbers are
  the causal ones.
- The error over all frames (3.60e-3) is dominated by the gripper joint, so `ratio` weights the gripper most;
  `effect` weights the joints equally.
- The napkin mask is a brightness threshold on 32 px patches.

Evidence: `artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/attention_heads_20261125/` (`heads.json`,
`block.json`, the figures; `maps.npz` is not committed).
