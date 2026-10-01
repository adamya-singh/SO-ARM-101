# Research log

One short entry per experiment, in order, written for a new MLE intern. Each
one says what was going wrong and how we knew, what we tried and what each
result would have meant, what happened, and the lesson. The running lab
notebook with every table is `notes/vision-rung-notebook.md`.

## August: rebuilding from a proven simulator

| # | question | answer in one line |
|---|---|---|
| [01](01-privileged-preflight.md) | Can the task even be solved? | No: the sim made the grasp impossible; new arm model, 15/15 |
| [02](02-oracle-clone.md) | Can a small network copy the cheating script? | On paper yes, in the sim 0/3: errors compound off the recorded path |
| [03](03-recovery-anchors.md) | Can a few recovery examples teach it to recover? | No, 8 examples made it worse (0/15) |
| [04](04-recovery-weights.md) | What if they count for less? | No weight worked; my static safety check was useless |
| [05](05-more-inputs.md) | Is it missing information? | Velocity, contact and history didn't help |
| [06](06-dagger.md) | What if it learns from its own mistakes? | DAgger capture worked, the small network couldn't fit it |
| [07](07-action-chunks.md) | What if it decides less often? | 90-step chunks: first learned grasp, then impossible commands |
| [08](08-first-promoted-policy.md) | How do we stop impossible commands? | Noise plus a feasibility penalty: first passing policy, 3/3 |
| [09](09-mujoco-version.md) | Why did the same run give different numbers? | A stray pip install swapped MuJoCo versions mid-project |
| [10](10-broader-evaluation.md) | Is it a policy or a recording? | A recording: 1.5 mm broke it; 5-start retrain almost works |
| [11](11-steps-and-width.md) | Do more steps or width finish the job? | 9/15; the failures were all the gripper past the recording's end |
| [12](12-learning-rate-schedule.md) | Does a better LR schedule fix it? | Cosine decay: 12/15, but seeds scatter 12/12/3 |
| [13](13-horizon-alignment.md) | What if the recording covers the whole task? | Fixed a problem already gone; real cause was gripper over-squeeze |
| [14](14-gripper-clamp.md) | Can we just clip the gripper? | Yes: first 15/15 with zero safety frames, on 2 of 3 seeds |
| [15](15-vision-v0.md) | Can it do this from pixels? | Badly (0-3/15), and it mostly ignored the image |
| [16](16-random-positions.md) | Does it work on unseen cube positions? | State 18/30, vision 0/30 |
| [17](17-data-scaling.md) | Is it a data problem? | State 30/30 at 400 positions; vision 24-30/30 with steps scaled |

## September: the real bench and the real arm

| # | question | answer in one line |
|---|---|---|
| [18](18-bench-teacher-and-detector.md) | Can the script do the bench task? | Yes after fixing the grasp checker's axis |
| [19](19-camera-mismatch.md) | Does the sim camera match the real one? | No: square 4x smaller, 300 px lower |
| [20](20-joint-map.md) | Do the sim's joint angles match the real arm? | No: wrist roll a quarter turn off, others up to 30 degrees |
| [21](21-lens-and-zeros.md) | What lens does the camera have? | 44 degrees, not 72; zeros corrected with a phone checkerboard |
| [22](22-first-bench-run.md) | Does the recipe learn the bench task from pixels? | 27/30, and it really uses the camera |
| [23](23-lens-matched-sim.md) | Can the sim see through the real lens? | Yes: 30/30, no undistortion needed |
| [24](24-first-real-attempts.md) | Does it work on the real arm? | No: the real image's look threw it off |
| [25](25-appearance-randomization.md) | Can it handle how the real scene looks? | Yes with randomized looks: passes real frames |
| [26](26-live-misses.md) | Why does it miss the real cube? | The cube was 20-40 mm off; a safety rule froze the arm |
| [27](27-first-live-success.md) | Does it work now? | Yes, once: grasp, 18.6 mm lift, back within 3 mm |
| [28](28-repeats-miss.md) | Can it do it again? | No: it goes to a learned spot, not to the cube |
| [29](29-placement-run-4.md) | Can it find a cube anywhere? | 3/30: memorized 400 placements (136x gap) |
| [30](30-placement-run-5.md) | Does 3x the placements fix it? | No: same 3/30, gap 417x |
| [31](31-scaling-ladder.md) | Do data, size or training length fix it? | No, and a lookup table beat every network |
| [32](32-random-shift.md) | Can we make memorizing impossible? | Random shifts: gap 5250x to 15x, 0/30 to 9/30 |
| [33](33-steps-vs-placements.md) | Now, do steps or placements help? | Steps re-memorize; placements pay, 15/30 |
| [34](34-seed-repeats.md) | How much of 15/30 is luck? | A lot: 15, 6, 12 across seeds |
| [35](35-placements-4800.md) | Does doubling placements again help? | Memorization gone, 6/30, and I misread why |
| [36](36-closure-in-mm.md) | Why does it still miss? | 10 mm scatter vs 8 mm tolerance; the second look fixes nothing |
