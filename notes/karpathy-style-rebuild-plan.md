# Diagnostics-First Rebuild Plan for SO-ARM-101 Robot Learning

## Document status

This is a living research and engineering plan for rebuilding the SO-ARM-101
learning stack from a small, trusted skeleton. It is inspired by the process in
Andrej Karpathy's *A Recipe for Training Neural Networks*: understand the data,
establish a correct end-to-end skeleton, prove that simple cases work, add
capacity until the system can fit the problem, regularize only after that, and
introduce complexity one justified step at a time.

This document is intentionally detailed about **how we should reason and what
evidence we should demand**, but deliberately flexible about exact model
architectures, dataset split ratios, thresholds, schedules, directory names,
and experiment counts. Those details should be selected from evidence available
at the relevant stage. They should not be frozen prematurely merely because an
early planning document happened to name one plausible value.

The plan is therefore a decision framework, not a rigid implementation
specification. We should revise it when experiments teach us something. When we
depart from it, the important requirement is to record why the change is
reasonable and what observation motivated it.

## Progress status (updated 2026-08-01)

- Phases 1-3 (task/eval contract, data understanding, coordinate/timing
  contracts): implemented in `src/so_arm101_v2/` with pinned resources and
  tests.
- Phases 5-6 (dumb baselines, tiny-model overfit gate): completed; artifacts
  under `artifacts/so_arm101_v2/`.
- **Phase 4 (prove the simulator): PASSED 2026-08-01.** The first preflight
  run exposed that a strict face grasp was geometrically impossible in every
  prior simulator configuration
  (`notes/privileged-controller-preflight-findings.md`); the v2 lane was then
  migrated to the MuJoCo Menagerie trs_so_arm100 model and the privileged
  controller now proves all 15 scenario rollouts deterministically with zero
  safety violations (`notes/menagerie-model-migration.md`,
  `artifacts/so_arm101_v2/simulation/preflight/`). This phase working exactly
  as designed - failing loudly before any learning - is the strongest
  validation of the plan's approach so far.
- Next per this plan: Phase 7+ (closed-loop model selection, stronger IL) on
  the now-proven simulator; the passing privileged controller doubles as a
  demonstration generator for supervised distillation.

---

## 1. Purpose

The project has already produced substantial infrastructure, experiments, and
hard-won debugging knowledge across physical data collection, MuJoCo simulation,
ACT imitation learning, SmolVLA, ReinFlow-style fine-tuning, PPO, coordinate
conversion, reward design, and physical deployment. The rebuild is not an
attempt to erase that work. Its purpose is to convert the strongest lessons
from that work into a much smaller system whose behavior we can explain and
trust from end to end.

The central objective is:

> Build a reproducible robot-learning pipeline in which data, coordinate
> systems, simulator behavior, model outputs, evaluation results, and physical
> commands are all independently inspectable and validated before advanced
> optimization is introduced.

The desired end state is not simply a cleaner codebase. It is a stronger
scientific instrument. When a model improves or fails, we should be able to say
which hypothesis was tested, which variables changed, which evidence supports
the conclusion, and which alternative explanations remain.

The rebuild should make it easy to answer questions such as:

- What exactly does one action value mean at every boundary in the system?
- Is the policy learning from the image, the robot state, both, or neither?
- Can the environment be solved by a controller with privileged information?
- Can a small model memorize a tiny subset of the demonstrations?
- Does offline prediction quality correlate with closed-loop task behavior?
- Are validation improvements robust across fixed evaluation scenarios?
- Does a simulation result transfer to physical observations, and where does
  the distribution shift appear?
- Is RL improving task behavior, or merely exploiting reward structure?
- Can another person reproduce a reported result without knowing the history of
  the repository?

---

## 2. Why a rebuild is warranted

The current repository contains valuable capabilities, but it has accumulated
many interacting sources of complexity:

- Multiple policy families and adaptation strategies.
- Multiple coordinate conventions and normalization paths.
- Large training scripts that combine policy construction, environment logic,
  rollout collection, optimization, checkpointing, evaluation, and logging.
- Several generations of reward functions and curricula.
- Old and new default checkpoints referenced by different tools.
- Large generated output trees located inside the working repository.
- Tests that currently depend on launch location or external artifacts.
- Qualitative observations, training metrics, and controlled evaluation results
  that are not always cleanly separated.

The physical ACT dataset itself contains an especially important contract:
same-frame `action` and `observation.state` are equal. This does not necessarily
make the dataset invalid, because future trajectory targets can still be
constructed from later frames. It does mean that target construction, action
lead, chunking, execution semantics, and coordinate conversion must be explicit
and visible rather than inherited indirectly through framework behavior.

The existing coordinate-contract work is an example of what the rebuild should
preserve. A systematic clipping failure was eventually traced to treating a
calibrated motor-range encoding as direct MuJoCo mechanical radians. The fix is
now supported by endpoint, round-trip, differentiability, joint-order, and
calibration-integrity tests. That style of explicit contract should become the
norm throughout the project.

The rebuild should also preserve negative results. Failed runs are useful when
they identify a disproven assumption, a confound, or an invalid experimental
path. They become harmful only when they remain mixed into the active system in
a way that makes current behavior ambiguous.

---

## 3. Governing principles

### 3.1 Earn complexity

Every major source of complexity should answer a demonstrated limitation of a
simpler system. We should not begin with the most powerful or fashionable
method available. We should begin with the smallest method that can answer the
current question.

Examples:

- Do not add action chunks until single-step or short-horizon targets have been
  validated.
- Do not add a variational objective until deterministic behavior cloning is
  understood.
- Do not add PPO until a supervised policy has a stable, measurable failure
  mode and the environment is independently known to be learnable.
- Do not add complex reward shaping until strict success and intermediate
  diagnostics are clearly separated.
- Do not add more cameras until the wrist-only observation path is measured and
  found insufficient for a specific reason.

### 3.2 One hypothesis per meaningful change

Experiments should be interpretable. A useful experiment has:

- A question.
- A predicted outcome.
- A controlled change.
- A predefined evaluation method.
- A result.
- A conclusion that distinguishes observation from interpretation.

Some mechanical changes naturally need to land together, but we should resist
bundling architecture, target definition, data augmentation, optimizer,
learning rate, reward, and evaluation changes into one run.

### 3.3 Treat visualizations as primary debugging tools

Numbers alone are not enough for this project. We should routinely visualize:

- Raw dataset frames and sequences.
- The exact tensors entering a model.
- The target trajectory paired with each input.
- Coordinate conversions joint by joint.
- Fixed model predictions throughout training.
- Closed-loop rollouts from fixed reset states.
- Contact geometry and grasp classification.
- Failure cases sorted by diagnostic categories.

Visualizations should be produced by normal tools in the pipeline, not by
one-off notebook code that becomes disconnected from the actual input path.

### 3.4 Separate correctness metrics from optimization metrics

Training loss, KL divergence, entropy, clip fraction, and gradient norms tell us
about optimization. They do not prove task correctness.

Task correctness should be measured separately through outcomes such as:

- Reaching the relevant workspace region.
- Making valid contact.
- Establishing a valid bilateral grasp.
- Lifting the object.
- Holding or transporting it as required.
- Completing the defined task without a safety violation.

Reward is also not equivalent to correctness. A policy can improve reward while
becoming worse at the real task.

### 3.5 Prefer reproducible comparisons over memorable anecdotes

Live simulation and physical observation remain valuable. They often reveal
failure modes that aggregate metrics hide. However, a selected result should
ultimately be supported by a repeatable evaluation suite with fixed scenario
identities, recorded outputs, and paired comparisons when possible.

We should label evidence honestly:

- **Diagnostic:** useful for debugging but not a performance claim.
- **Qualitative:** based on inspection of behavior or videos.
- **Provisional:** measured, but with limited scenarios or unresolved confounds.
- **Controlled:** evaluated under a predefined and repeatable protocol.
- **Final:** evaluated on a held-out test protocol after model selection.

### 3.6 Preserve provenance

Every meaningful artifact should be traceable to:

- Code revision or source snapshot.
- Fully resolved configuration.
- Dataset version and split manifest.
- Calibration version.
- Simulator model version.
- Random seed and evaluation scenario IDs.
- Dependency environment.
- Parent checkpoint, if any.

The goal is not bureaucratic completeness. The goal is to make comparison and
reproduction possible months later.

### 3.7 Keep the plan adaptable

Decision gates in this plan are not promises to use a particular numerical
threshold forever. Their purpose is to prevent us from moving on while basic
questions remain unanswered. The most appropriate threshold may depend on the
noise level, task definition, available data, hardware, or observed variance.

When a gate must change, we should revise it before looking at the final result
of the experiment that will be judged by it, whenever practical.

---

## 4. High-level research ladder

The default sequence is:

```text
Problem and success contract
    ↓
Data understanding and immutable data versions
    ↓
Coordinate, timing, and action contracts
    ↓
Simulator and physical-interface validation
    ↓
Trivial and privileged baselines
    ↓
Tiny supervised models and overfit tests
    ↓
Closed-loop behavior cloning evaluation
    ↓
More capable imitation-learning policies
    ↓
Controlled physical deployment
    ↓
RL or other post-training methods, if evidence justifies them
```

This sequence is a default, not a ban on revisiting earlier stages. In practice,
we should expect loops. For example, closed-loop evaluation may reveal that a
dataset lacks recovery behavior, sending us back to data collection. Physical
deployment may reveal a camera shift, sending us back to observation auditing.

---

## 5. Phase 0: Preserve the current project and establish a clean lane

### Goal

Create a safe place to rebuild without destroying current experiments or
allowing historical complexity to leak invisibly into the new path.

### Work

- Preserve the existing repository state and uncommitted work before large
  restructuring begins.
- Treat current notes, outputs, checkpoints, and scripts as a research archive
  and source of hypotheses.
- Create a clearly named v2 package or top-level implementation area.
- Avoid moving or deleting historical files during the first rebuild stages.
- Define ignore rules and artifact-storage conventions so future runs do not
  add tens of gigabytes of generated data to normal source-control workflows.
- Record which legacy modules are considered authoritative for specific known
  contracts, such as coordinate conversion or physical calibration.

### Design preference

The new code should be a proper Python package so tools and tests run from the
repository root. The exact package name and directory layout may change, but
the separation of concerns should remain clear.

A plausible starting shape is:

```text
src/<package>/
    contracts/
    data/
    simulation/
    policies/
    training/
    evaluation/
    deployment/

tools/
configs/
tests/
docs/ or notes/
```

We should avoid copying large legacy modules wholesale. When reusing behavior,
first identify the smallest relevant unit, write or migrate tests for its
contract, and then port it.

### Exit evidence

- Root-level import and test commands work without changing directories.
- Generated artifacts are separated from source.
- A new contributor can identify the active v2 path and the legacy archive.
- The rebuild can proceed without modifying historical experimental evidence.

---

## 6. Phase 1: Define the task and evaluation contract

### Goal

State exactly what behavior constitutes success before optimizing a policy.

### Initial scope

Begin with one deliberately narrow manipulation task. A reasonable initial
candidate is a fixed or tightly controlled cube pickup using the wrist camera
and robot joint state. The exact cube pose distribution, lift requirement,
episode duration, and hold duration should be selected after inspecting the
physical data and simulator geometry.

The initial task definition should include:

- Starting robot-state distribution.
- Object identity, geometry, and allowed pose distribution.
- Observation keys and their semantic meaning.
- Control frequency and action execution semantics.
- Episode termination rules.
- Safety bounds.
- Strict success definition.
- Intermediate diagnostic events.

### Success versus reward

Implement success detection independently from the training reward. Success
should correspond to the task we would recognize as completed, not simply to a
large accumulated score.

Intermediate measurements may include:

- Distance or alignment to a useful pre-grasp region.
- First contact.
- Contact location and normal quality.
- Bilateral contact.
- Grasp persistence.
- Object displacement.
- Object height change.
- Sustained lift.
- Drop, slip, or unsafe contact.

These measurements are diagnostic. They can later inform reward design, but
they should exist even when no shaped reward is used.

### Evaluation suite structure

Maintain named evaluation suites rather than a single vague `--episodes N` run.
Potential suites include:

- A deterministic fixed-reset debugging suite.
- A small set of nearby perturbations.
- Appearance or lighting variations.
- A broader generalization suite.
- A held-out physical protocol.

Not every suite must exist immediately. Add them as the scope expands. Each
suite should have stable scenario identities so two policies can be evaluated
on matched conditions.

### Exit evidence

- The task contract is machine-readable and documented.
- Success can be computed without consulting the reward value.
- Known invalid behaviors do not count as success.
- Evaluation scenarios can be replayed by identity.
- A video and telemetry record can explain why a rollout passed or failed.

---

## 7. Phase 2: Become one with the datasets

### Goal

Understand what the data actually contains before choosing a model or target
representation.

### Dataset inventory

Create an inventory of every dataset that might influence the rebuild:

- Physical demonstrations.
- Fixed-block simulation demonstrations.
- Randomized-block simulation demonstrations.
- Any failed, partial, recovery, or auxiliary recordings.
- Calibration and metadata files associated with collection.

For each dataset, record:

- Source and collection process.
- Episode and frame counts.
- Camera keys, image sizes, frame rates, and codecs.
- State and action shapes.
- Joint order and coordinate encoding.
- Task instructions.
- Known changes in hardware, camera mount, lighting, or calibration.
- Whether outcome labels exist.
- Whether demonstrations are all successful.

### Visual audit

Build a repeatable tool that generates contact sheets and short clips. It
should make it easy to inspect:

- Random episodes.
- Every episode start and end.
- Representative frames at normalized progress points.
- Shortest and longest episodes.
- Extreme joint states and actions.
- Fast-motion and near-stationary segments.
- Gripper opening and closing transitions.
- Dark, bright, blurry, frozen, or corrupted frames.
- Examples with unusual backgrounds or object locations.

The visualization should be constructed from the same decoded records and
preprocessing code that training will use. We should also provide a view of the
final model tensor after cropping, resizing, channel conversion, normalization,
and augmentation.

### Statistical audit

Measure at least:

- Episode-length distribution.
- Per-joint state and target ranges.
- Per-joint velocities and accelerations.
- Frequency of values near encoding or mechanical limits.
- Gripper state distribution and transition frequency.
- Starting and ending pose distributions.
- Timestamp monotonicity and frame-interval distribution.
- Repeated or missing frames.
- Duplicate or near-duplicate episodes where practical.
- Image-channel statistics and obvious camera discontinuities.

The existing fact that same-frame action equals same-frame state should become
a formal invariant in the dataset report, along with the distribution of
future deltas at several candidate horizons.

### Outcome and phase labels

If feasible, annotate each episode with coarse outcome and phase information:

- Full success.
- Partial lift.
- Valid grasp without lift.
- Contact without grasp.
- Miss or unrelated motion.
- Unsafe or unusable demonstration.

Exact frame-level phase annotation may be unnecessary at first. Episode-level
labels and a few automatically detected events can already answer important
questions about dataset quality and imbalance.

### Splitting strategy

Create immutable split manifests at the episode level. Adjacent video frames
from one episode must not be distributed across training and evaluation splits.

The exact ratio should depend on dataset size, condition diversity, and the
amount of variance observed. The split should consider stratification by any
important source of variation, such as:

- Collection session.
- Initial object pose.
- Operator behavior.
- Success or failure category.
- Hardware or camera configuration.

Keep a validation split for model selection and a test split for final
assessment. If the data is too small for a conventional held-out test to be
informative, consider repeated episode-level resampling or cross-validation,
but preserve a genuinely untouched final protocol for important claims.

### Data versioning

A dataset version should be defined by a manifest and content hashes rather
than by a mutable directory name alone. Derived targets and filtered subsets
should record their source dataset and transformation configuration.

### Exit evidence

- We have watched and categorized the data rather than relying only on metadata.
- Dataset statistics and outliers are reproducibly generated.
- Every training sample can be traced to an episode and source version.
- Split manifests are stable and episode-based.
- We can explain what information is and is not present in the demonstrations.
- We know whether the data supports the proposed initial task.

---

## 8. Phase 3: Make coordinate, timing, and action contracts explicit

### Goal

Ensure that every value has one documented meaning at every system boundary.

### Coordinate domains

Define named types or validated structures for distinct domains, including as
appropriate:

- Raw servo units.
- Calibrated LeRobot motor ranges.
- Dataset-encoded state.
- Dataset-encoded target/action.
- Model-normalized state.
- Model-normalized output.
- MuJoCo mechanical joint coordinates.
- Physical deployment commands.

Avoid passing an unlabelled six-element array through multiple layers and
assuming callers remember which domain it represents. Runtime validation,
shape checks, explicit function names, and typed wrappers are preferable.

### Timing contract

Document:

- Camera sampling frequency.
- State sampling frequency.
- Dataset timestamps.
- Model observation time.
- Target time or trajectory window.
- Action execution rate.
- Whether actions are repeated, interpolated, or temporally aggregated.
- Any physical communication latency.

The same-frame action/state equality makes timing especially important. We
should build future targets explicitly and inspect them before training.

### Candidate target representations

Potential targets include:

- Future absolute joint pose.
- Future delta from the current pose.
- Velocity-like command over a known interval.
- Short trajectory chunk.

We should begin with the representation that is easiest to verify and execute
correctly. More sophisticated targets can be compared later. The plan does not
assume in advance that absolute targets, deltas, or chunks will be best.

### Contract visualizer

For any selected sample, show:

- The observation image.
- Current dataset state.
- Selected future target or trajectory.
- Target converted into MuJoCo coordinates.
- Target converted into physical command coordinates where safe and relevant.
- Per-joint limits and clipping indicators.
- The observed trajectory between the current frame and target frames.

### Tests

Preserve and expand the current coordinate tests:

- Joint ordering.
- Endpoint mapping.
- Midpoint mapping.
- Round trips for individual and batched values.
- Calibration-file integrity.
- Representative historically problematic values.
- Differentiability where training requires it.
- Clipping and out-of-range behavior.
- Shape, dtype, and device handling.

Add timing and target tests:

- Targets do not cross episode boundaries.
- Lead and chunk indices match intended timestamps.
- Padding behavior is explicit.
- Terminal frames are handled consistently.
- Training and inference execute the same target semantics.

### Exit evidence

- Every boundary conversion is named, tested, and inspectable.
- Representative dataset targets can be executed in simulation without hidden
  unit changes or systematic clipping.
- Training and inference share one action contract.
- Off-by-one and episode-boundary behavior is covered by tests.

---

## 9. Phase 4: Prove the simulator and measurement system

### Goal

Show that the environment, dynamics, contacts, resets, cameras, and outcome
detectors are coherent before asking a learned policy to solve the task.

### Simulator contract

Validate:

- MJCF joint names, ordering, limits, actuators, and control ranges.
- Reset poses and object placements.
- Camera identity, pose, field of view, orientation, and render preprocessing.
- Simulation step size and action-control interval.
- Contact geom identities and relevant collision settings.
- Object mass, friction, dimensions, and table geometry.
- Termination and truncation behavior.

### Golden states and images

Store a small number of reference reset states and rendered images. These are
not intended to prevent all future simulator changes. They make changes visible.
When the camera or model is deliberately updated, regenerate the references in
a reviewed change with an explanation.

### Scripted or privileged controller

Develop the simplest controller that can solve the narrow task using privileged
state. It may be scripted inverse kinematics, waypoint control, a state-space
policy, or another method appropriate to the simulator.

The controller is not a deployment candidate. It is an environment test. If a
controller with direct access to cube and gripper state cannot solve the fixed
task, visual learning should not begin.

The controller should exercise:

- Approach.
- Alignment.
- Gripper closure.
- Contact and grasp detection.
- Lift.
- Success and termination.

### Negative tests

Construct trajectories or states that should not count as success:

- Hovering near the cube.
- One-sided contact.
- Pushing the cube.
- Corner pinching that fails the intended grasp definition.
- Lifting through an invalid reset or simulator artifact.
- Brief height spikes without a sustained grasp.
- Unsafe airborne or floor-assisted contact.

### Reward validation

At this stage, reward should be treated as an inspected measurement, not a
trusted training objective. For scripted trajectories, plot every reward
component and diagnostic event over time. Confirm that rewards increase for the
intended reasons and that no single component dominates through an accidental
scale or repeated event.

### Exit evidence

- Fixed resets are reproducible.
- A privileged controller reliably demonstrates valid success.
- Known invalid behaviors are rejected.
- Camera and state outputs have stable, documented semantics.
- Reward and success traces can be explained frame by frame.

---

## 10. Phase 5: Establish dumb baselines

### Goal

Measure how much performance is available without meaningful visual learning.

### Candidate baselines

Use whichever baselines are applicable to the chosen target contract:

- Hold the current pose.
- Predict a constant mean or median target.
- Predict an initial-pose-dependent constant.
- Repeat the previous command.
- Use only robot state.
- Use only the image.
- Zero or shuffle one input modality.
- Retrieve a nearest-neighbor trajectory.
- Train a linear or shallow regressor.
- Train a privileged-state policy in simulation.

These baselines answer different questions. An input-independent learned bias
tests whether class or trajectory imbalance explains apparent performance. A
state-only model tests whether the wrist image contributes useful information.
A nearest-neighbor policy tests whether the task is mostly memorization.

### Offline and closed-loop evaluation

Where possible, evaluate baselines both offline and in closed loop. A constant
pose can have moderate offline error while being useless as a controller. A
slightly worse offline predictor may be more stable during execution.

### Exit evidence

- We know the performance floor for trivial behavior.
- We know whether each input modality contains measurable signal.
- Future learned models must beat relevant baselines on predefined metrics.
- The privileged simulation baseline supports the learnability of the task.

---

## 11. Phase 6: Build the smallest supervised learning skeleton

### Goal

Create a complete training and evaluation loop whose behavior is easy to
inspect and whose failures are localizable.

### Minimal model

Start with a deliberately small model appropriate to the selected inputs. For
example, a small image encoder and a small state encoder may be combined to
predict one future pose. The exact architecture is less important than:

- It is small enough to train quickly.
- Its forward path is easy to inspect.
- It has enough capacity to memorize a tiny batch.
- It uses the same preprocessing and target construction intended for later
  models.

Avoid optional complexity at first:

- No augmentation unless required to make inputs well formed.
- No elaborate learning-rate schedule.
- No stochastic latent variable.
- No auxiliary losses.
- No long action chunks.
- No pretrained foundation model unless a tiny scratch model has already
  established the pipeline.

### Initialization checks

Before training:

- Inspect predicted target distributions.
- Check that the initial loss is plausible for the target scaling.
- Initialize output biases from target statistics when appropriate.
- Verify that gradients reach the expected modules.
- Check for nonfinite values.
- Confirm that a loss based on one batch example does not depend on another
  example unexpectedly.

### Overfit ladder

Progress through increasingly demanding memorization tests:

1. A few hand-selected samples.
2. One short trajectory or episode.
3. A small collection of episodes.
4. The full training split.

At the smallest scale, visualize labels and predictions together throughout
training. Failure to fit a tiny dataset should stop the project until its cause
is understood.

### Fixed prediction batch

Maintain a fixed diagnostic batch containing representative and difficult
examples. At intervals, record:

- Predicted trajectories.
- Per-joint errors.
- Output variance.
- Sensitivity to image and state changes.
- Example-level loss ranking.

### Metrics

Do not rely only on a normalized aggregate loss. Report metrics in meaningful
domains, potentially including:

- Per-joint dataset-space error.
- Per-joint mechanical-coordinate error.
- Endpoint error at the executed horizon.
- Trajectory shape error.
- Gripper timing error.
- Limit or clipping frequency.
- Validation error by episode category.

### Exit evidence

- The model can memorize a tiny sample set.
- Increasing capacity lowers training error when expected.
- Removing or shuffling useful input makes performance worse.
- Training is reproducible enough for debugging.
- Full-split performance beats relevant trivial baselines.

---

## 12. Phase 7: Make closed-loop evaluation the center of model selection

### Goal

Determine whether offline learning produces useful behavior when the policy's
own actions change future observations.

### Why this matters

Behavior cloning is vulnerable to compounding error. A model can predict held-
out demonstration targets accurately while drifting into states absent from
the demonstrations. Offline loss is necessary evidence, but it is not the
primary task result.

### Evaluation protocol

For each candidate:

- Use the same named reset scenarios.
- Use deterministic inference where appropriate for the comparison.
- Record per-episode metrics and videos.
- Report aggregate outcomes with uncertainty.
- Compare paired outcomes rather than only unrelated averages.
- Record clipping, saturation, and safety interventions.

Potential evaluation layers:

- Training-distribution replay diagnostics.
- Fixed-condition simulation.
- Nearby state perturbations.
- Appearance perturbations.
- Wider task variation.
- Physical evaluation.

We should not require all layers before making any progress. Start with the
smallest suite that can reveal closed-loop behavior, then expand as policies
become competent.

### Failure taxonomy

Categorize failures rather than reducing everything to one return:

- No meaningful motion.
- Incorrect initial direction.
- Stops short of the object.
- Misses laterally or vertically.
- Ground or table collision.
- Reaches but closes too early or too late.
- Pushes rather than grasps.
- Grasp slips.
- Grasps but fails to lift.
- Lifts but fails to sustain or complete.
- Oscillation or unstable chunk execution.

This taxonomy should guide the next experiment. For example, consistent late
closure may suggest target timing or demonstration alignment, while broad
visual misses may suggest observation shift or insufficient image features.

### Exit evidence

- Model selection uses controlled closed-loop evidence.
- Offline metrics and task behavior are reported separately.
- Failure modes are reproducible and categorized.
- Videos and telemetry support aggregate claims.

---

## 13. Phase 8: Introduce stronger imitation-learning models

### Goal

Add model capacity and temporal structure in response to measured limitations
of the minimal policy.

### Possible progression

Depending on results, candidate additions may include:

- A stronger image backbone.
- Pretrained visual features.
- A short deterministic action trajectory.
- Action chunking.
- Temporal aggregation.
- ACT without or with its latent-variable components.
- Multiple observation frames.
- Additional cameras.
- Language or task conditioning.

This list is not an ordering mandate. The correct next step depends on the
failure mode. If the minimal model cannot localize the cube, improve visual
representation. If it reaches correctly but executes poorly, investigate
target timing and control. If behavior is multimodal, a latent policy may be
justified. If recovery is absent from the data, a larger architecture may not
help at all.

### ACT-specific questions

Before trusting ACT, explicitly test:

- What does each chunk element predict?
- Which chunk elements are executed?
- How are overlapping chunks combined?
- How does action lead interact with chunk construction?
- Does padding bias terminal targets?
- Is the VAE being used, and what happens when it is disabled?
- How sensitive is behavior to sampling versus deterministic decoding?
- Does the policy overfit one episode and reproduce its trajectory?
- Which checkpoint has the best closed-loop validation behavior, rather than
  merely the lowest loss?

### Pretraining and augmentation

Pretraining and augmentation are strong tools, but they should be introduced
with explicit hypotheses. An augmentation must preserve the task label or
transform it consistently. For a wrist-mounted camera, geometric augmentation
can alter the apparent relationship between image and joint action, so it
should not be assumed harmless.

### Exit evidence

- Each added capability addresses an observed limitation.
- Comparisons isolate the important change as much as practical.
- Stronger models are judged by closed-loop validation behavior.
- We retain the simplest model whose performance is sufficient.

---

## 14. Phase 9: Improve the data when the model exposes data limitations

### Goal

Use model failures to guide targeted data collection rather than repeatedly
tuning around missing coverage.

### Data-centric questions

For each common failure, ask:

- Does the dataset contain the desired recovery behavior?
- Does it contain enough examples near this state?
- Are grasp timing and gripper transitions represented consistently?
- Are trajectories smooth and purposeful?
- Are demonstrations successful under the exact task contract?
- Is there conflicting supervision for visually similar states?
- Is the observation sufficient to infer the correct action?

### Targeted collection

Potential additions include:

- More examples of the dominant failure state.
- Recovery trajectories from off-nominal approaches.
- Balanced initial cube positions.
- Cleaner grasp and lift demonstrations.
- Slower or more consistent demonstrations around contact.
- Additional camera or calibration sessions when distribution shift is proven.

New data should be versioned separately first. Compare old-only, new-only, and
combined training where useful. Avoid silently replacing the dataset and
making prior results impossible to interpret.

### Exit evidence

- New collection is motivated by a measured coverage gap.
- Dataset versions and provenance remain clear.
- Added data improves the targeted validation failure mode.
- Improvements do not merely shift failures elsewhere unnoticed.

---

## 15. Phase 10: Physical deployment and sim-to-real evaluation

### Goal

Evaluate physical behavior safely while treating simulation and reality as
related but distinct domains.

### Key principle

A policy trained on physical demonstrations should first be assessed on held-
out physical data. Simulation can provide valuable closed-loop testing, but a
simulation camera is not automatically a valid substitute for a physical wrist
camera. Conversely, success in simulation does not establish physical success.

### Preflight stages

Progress through increasingly active checks:

- Load and validate calibration and checkpoint artifacts.
- Run a fake-observation model smoke test.
- Process live observations without moving the robot.
- Log proposed actions and verify their ranges, velocities, and temporal
  behavior.
- Test conservative motion away from the object.
- Run task trials with reduced speed or bounded action changes.
- Expand only after reviewing telemetry and video.

### Safety layer

The deployment path should enforce safety independently from the learned
policy. Consider:

- Mechanical and calibrated joint bounds.
- Velocity or per-step change limits.
- Workspace constraints where feasible.
- Communication timeout behavior.
- Emergency stop access.
- Watchdog and maximum episode duration.
- Termination on repeated clipping or implausible commands.
- Explicit confirmation before enabling motor motion.

The exact limits should come from hardware capabilities and calibration, not
arbitrary values in this document.

### Physical evaluation protocol

Define repeatable reset procedures and record every trial. Report:

- Trial identity and reset condition.
- Success and intermediate outcomes.
- Safety intervention or clipping.
- Video.
- State and commanded-action logs.
- Calibration and checkpoint hashes.

Qualitative user inspection remains valuable and should be recorded, but it
should be paired with a controlled trial table.

### Exit evidence

- The policy passes inference and action preflights before motion.
- Safety constraints are independent and tested.
- Physical results use a repeatable protocol.
- Sim-to-real differences are measured rather than assumed.

---

## 16. Phase 11: Introduce reinforcement learning only when justified

### Goal

Use RL to improve a specific, measured limitation after supervised behavior and
environment correctness have been established.

### Preconditions

Before returning to PPO, ReinFlow, or another post-training method, we should
have evidence that:

- The environment can be solved by a privileged controller or baseline.
- Success and reward instrumentation are trustworthy.
- The supervised policy exhibits repeatable useful behavior.
- The remaining failure is plausibly correctable through interaction.
- The action distribution and log-probability definition are coherent.
- Evaluation is separate from training and uses fixed scenarios.

### Start simple

The first RL experiment should minimize degrees of freedom. Potential choices
include:

- Fixed task and reset distribution.
- Small trainable policy subset.
- Frozen image backbone initially.
- Conservative update sizes.
- Sparse success with only well-audited shaping.
- Short runs designed to validate optimization rather than achieve peak
  performance.

### PPO and stochastic-policy invariants

Where applicable, verify:

- Old and new policy evaluations match before any optimizer step.
- The policy evaluation path is deterministic when the algorithm assumes it.
- Log-probability scaling is understood across action dimensions.
- Batch examples do not mix unexpectedly.
- KL, ratios, entropy, clip fraction, and gradient norms are finite and
  interpretable.
- Actor and critic parameter scopes are explicit.
- Critic learning does not unintentionally change actor features.
- Checkpoint resume and branch semantics are tested.

### Reward discipline

Maintain a strict distinction among:

- Success metric.
- Diagnostic events.
- Training reward.

Plot reward components over actual rollouts. If a policy improves reward while
strict success or precursor behavior worsens, treat it as reward exploitation
until shown otherwise.

### Multi-seed evidence

One seed is suitable for debugging. It is not sufficient for a performance
claim. Once the system is stable, compare candidates across matched seeds and
evaluation scenarios. The number of seeds should reflect observed variance and
available compute rather than a fixed ritual.

### Exit evidence

- Optimization invariants pass before training.
- RL improves predefined closed-loop outcomes, not only reward.
- Results are reproducible across enough independent runs to support the claim.
- The reason for using RL remains stronger than collecting better data or using
  a simpler supervised correction.

---

## 17. Experiment management

### Experiment proposal template

Before a meaningful run, record:

```text
Question:
Hypothesis:
Motivation/evidence:
Controlled change:
Unchanged conditions:
Primary evaluation:
Secondary diagnostics:
Expected outcome:
Possible interpretations if it succeeds:
Possible interpretations if it fails:
Stopping or invalidation conditions:
```

This can be short for small experiments. The purpose is to avoid running first
and inventing the hypothesis afterward.

### Run manifest

Each retained run should include a compact, machine-readable manifest with:

- Run name and timestamp.
- Source revision and dirty-state indicator.
- Resolved configuration.
- Dataset manifest and split IDs.
- Simulator and calibration hashes.
- Parent checkpoint provenance.
- Seeds.
- Hardware and important dependency versions.
- Output artifact checksums or locations.
- Evaluation suite versions.

### Results record

After the run, add:

```text
Status:
Observed result:
Primary metrics:
Important diagnostics:
Qualitative observations:
Confounds or anomalies:
Conclusion:
Next decision:
```

### Naming

Names should identify the question or controlled change, not make a success
claim. Avoid names such as `final`, `best`, or `solved` until the evidence truly
supports them.

### Checkpoint selection

Select checkpoints using validation behavior under a predefined rule. Do not
choose the test winner after inspecting the test set. Retain intermediate
checkpoints when they help answer whether longer training improved or degraded
closed-loop behavior.

---

## 18. Evaluation and statistical reporting

### Primary metrics

The primary metric should correspond directly to the task contract, usually a
strict success rate. Depending on the stage, it may be accompanied by:

- Confidence intervals or another uncertainty estimate.
- Per-scenario outcomes.
- Paired differences between policies.
- Time-to-success or steps-to-success.

### Secondary metrics

Use intermediate outcomes to explain changes:

- Reach or alignment quality.
- Contact rate.
- Valid grasp rate.
- Grasp persistence.
- Maximum and sustained lift.
- Final object displacement.
- Collision and safety rates.

These should not be combined into a single headline number unless there is a
clear reason.

### Optimization diagnostics

Track what is appropriate to the algorithm:

- Training and validation losses.
- Per-joint errors.
- Gradient and parameter norms.
- Learning rate.
- Prediction variance.
- KL and PPO ratio statistics.
- Entropy and log standard deviation.
- Throughput and resource use.

### Evaluation integrity

- Validation guides development.
- Test evaluation is infrequent and final.
- Scenarios are matched for comparisons.
- Failed or interrupted runs remain visible.
- Missing artifacts invalidate only the affected claim, not unrelated tests.
- Evaluation should not depend on mutable `last` symlinks without recording
  their resolved targets.

---

## 19. Testing strategy

### Unit tests

Cover pure and local behavior:

- Coordinate transforms.
- Target indexing.
- Normalization and denormalization.
- Reward components.
- Contact and grasp classification.
- Configuration resolution.
- Checkpoint metadata parsing.
- Metric aggregation.

### Integration tests

Cover boundaries:

- Dataset sample through preprocessing and model input.
- Model output through postprocessing and simulator action.
- Environment reset and one-step transition.
- Checkpoint save/load equivalence.
- Training and inference target semantics.
- Physical action preflight without motor movement.

### Golden tests

Use reviewed reference artifacts for:

- Reset observations.
- Camera orientation.
- Representative coordinate mappings.
- Known reward traces.
- A small deterministic rollout.

Golden tests should reveal meaningful changes without making intentional
improvements prohibitively difficult.

### Smoke tests

Provide cheap commands for:

- Import and dependency validation.
- Dataset metadata validation.
- A single forward and backward pass.
- A tiny overfit run.
- A short headless simulation rollout.
- A checkpoint inference pass.

### Artifact-independent tests

Core tests should not fail merely because a large optional checkpoint is absent.
Tests requiring external artifacts should declare and skip that dependency
clearly, or use a small fixture designed for the test.

---

## 20. Observability and visualization tools

The rebuild should make the following views easy to produce:

### Data explorer

- Episode browser.
- Synchronized video, state, action, and target plots.
- Search by ranges, outcome labels, and outlier scores.

### Model-input viewer

- Raw frame.
- Final resized/normalized tensor.
- Augmented view if augmentation is enabled.
- Current state and target in named coordinate domains.

### Prediction dynamics

- Fixed-example predictions across checkpoints.
- Per-joint predicted versus target trajectories.
- Uncertainty or sampling variation where relevant.

### Closed-loop dashboard

- Video.
- Robot and object state.
- Policy actions before and after safety filtering.
- Contact and grasp events.
- Reward components.
- Strict success state.

### Failure explorer

- Rollouts grouped by failure category.
- Worst and borderline examples.
- Comparisons of two policies on the same scenarios.

The initial implementation can be simple static HTML, images, videos, and JSON.
An interactive interface should be added only if it materially speeds analysis.

---

## 21. Configuration and dependency strategy

### Configuration

Use validated configuration objects with resolved values written to every run.
Defaults should be centralized rather than repeated across scripts. Configuration
should distinguish:

- Task and environment.
- Dataset and split.
- Target/action contract.
- Model.
- Training.
- Evaluation.
- Deployment and safety.

Avoid long shell commands as the sole source of experimental truth. Shell
scripts may launch campaigns, but each run should preserve a resolved config.

### Dependencies

Pin enough of the environment to reproduce important results while avoiding a
single enormous, accidental freeze as the only specification. Separate core,
simulation, training, visualization, and hardware extras where useful.

Record the exact runtime environment for retained runs. Tests should establish
which dependency combinations are supported.

### External frameworks and forked code

Treat the LeRobot fork as an explicit dependency with a known revision. Local
modifications should either be upstreamed into the fork with tests or isolated
behind a small adapter. Avoid keeping an unverified reference copy of framework
source as the effective patch mechanism.

---

## 22. Migration policy for legacy knowledge

The legacy stack contains important findings that should be translated into
tests and design constraints.

### Preserve immediately

- ACT/MuJoCo coordinate conversion and calibration integrity.
- Joint order and physical encoding documentation.
- PPO old/new log-probability self-consistency checks.
- Deterministic actor evaluation where required.
- Action-dimensionality awareness for stochastic policies.
- Strict grasp/contact diagnostics and known corner-contact failures.
- Evidence that training loss alone did not predict pickup success.

### Re-evaluate before preserving as defaults

- Particular action leads.
- Chunk lengths.
- Reward scales.
- Curriculum stage probabilities.
- Actor learning rates.
- Sigma bounds.
- Preferred checkpoints.
- Specific camera preprocessing choices.

These settings arose in a system with known confounds and should be treated as
hypotheses, not timeless truths.

### Archive rather than port

- Campaign launchers tied to obsolete checkpoints.
- Deprecated policy paths with no active evaluation use.
- Duplicated adapters superseded by a single tested interface.
- One-off debugging code whose lesson can be captured by a regression test.

---

## 23. Common failure modes this plan is designed to prevent

- Training a complex model before inspecting the dataset.
- Treating a framework's tensor shape compatibility as semantic correctness.
- Confusing calibrated motor encodings with mechanical coordinates.
- Using same-frame states as if they were future commands.
- Allowing action chunks to cross episode boundaries.
- Comparing models on different random resets.
- Selecting checkpoints by loss while ignoring closed-loop behavior.
- Tuning on the test set.
- Adding reward terms until return improves without validating task success.
- Interpreting one visually compelling rollout as a general result.
- Running multi-seed campaigns before one seed is mechanically correct.
- Assuming a fixed seed guarantees determinism across all operations.
- Changing camera geometry, coordinates, reward, and policy at the same time.
- Allowing missing large checkpoints to break the core unit-test suite.
- Keeping run outputs in the source tree without clear retention policy.
- Adding a larger model where the real limitation is missing recovery data.
- Using RL because imitation learning failed for an unexplained reason.

---

## 24. Initial implementation tranche

The first tranche should be deliberately bounded. Its goal is not to train the
final robot policy. Its goal is to establish the trusted skeleton.

### Suggested scope

1. Establish the v2 Python package and root-level test command.
2. Define the task, observation, action, and success contract structures.
3. Port the coordinate adapter and its high-value regression tests.
4. Add dataset manifests and episode-level split support.
5. Build the physical dataset audit and visualization report.
6. Implement explicit future-target construction with boundary tests.
7. Add trivial offline baselines.
8. Add a minimal supervised model and a tiny-batch overfit command.
9. Add a small, named simulation evaluation suite.
10. Write a baseline report summarizing what was learned before any long run.

### What this tranche should not require

- Reproducing every legacy policy.
- Training ACT to completion.
- Running PPO.
- Reimplementing every reward profile.
- Solving sim-to-real transfer.
- Cleaning or deleting the historical output tree.
- Choosing permanent hyperparameters.

### Tranche completion evidence

- Any selected training example can be visualized end to end.
- Its future target is correct and cannot cross an episode boundary.
- Coordinate conversions are tested in both directions.
- A tiny model can overfit a tiny sample.
- Trivial baselines have recorded results.
- A simulator reset and short rollout are reproducible.
- The entire core test suite runs from the repository root.

---

## 25. Decision gates and possible pivots

The following gates guide sequencing without prescribing exact thresholds.

### If data inspection reveals many failed or inconsistent demonstrations

Pause model work. Label, filter, or recollect data. Consider whether failures
should be retained for recovery learning under an explicit objective.

### If future targets are noisy or ambiguous

Compare target definitions, smooth trajectories where defensible, inspect
latency, or collect cleaner demonstrations. Do not immediately compensate with
a more expressive stochastic model.

### If the privileged controller cannot solve simulation

Fix the environment, controller authority, contacts, reset distribution, or
success detector before visual learning.

### If a tiny model cannot overfit a tiny batch

Investigate data loading, target construction, normalization, loss, gradients,
model capacity, and regularization. Do not proceed to ACT.

### If offline validation improves but closed-loop behavior does not

Investigate compounding error, execution semantics, target horizon, temporal
aggregation, observation shift, and missing recovery data.

### If state-only matches image-plus-state

Determine whether the task is too fixed, the image path is broken, or the image
contains no additional information. Expand task variation only when the base
pipeline is trustworthy.

### If simulation works but physical behavior fails

Measure camera, state, timing, action, dynamics, and calibration shift. Avoid
assuming that more simulated RL will solve an unmeasured observation mismatch.

### If RL reward improves without strict success

Treat it as a reward-design failure or exploitation signal. Inspect rollouts and
component traces before changing optimization hyperparameters.

### If additional complexity yields no reliable gain

Prefer the simpler system. Complexity is a cost in debugging, deployment, and
future experimentation even when inference speed is acceptable.

---

## 26. Open questions to resolve through the process

These questions should remain open until the relevant evidence is collected:

- Which subset of existing physical demonstrations are clean successes?
- How much variation exists in cube pose, lighting, and starting arm state?
- What future horizon best matches physical control latency and operator motion?
- Are absolute pose targets or deltas more stable in closed loop?
- Is one wrist frame sufficient, or is short observation history necessary?
- How closely do simulated wrist images match the physical camera after the
  coordinate path is fixed?
- Which simulator parameters most strongly affect grasp transfer?
- Does a deterministic policy capture the demonstrations adequately?
- Is action chunking helpful because of temporal coherence, or harmful because
  it reduces feedback frequency?
- Are current demonstrations sufficient for recovery from model-induced drift?
- Which intermediate metric best predicts eventual pickup success?
- Does ACT offer a meaningful gain over a small deterministic policy on this
  dataset?
- Is a pretrained visual backbone beneficial after controlled evaluation?
- If RL is used, which policy subset can be safely and effectively adapted?
- Would targeted new demonstrations provide more value than another post-
  training campaign?

The plan should help answer these questions rather than silently assuming their
answers.

---

## 27. Definition of project success

The rebuild is successful at multiple levels.

### Engineering success

- The active system is modular, tested, and runnable from a clean environment.
- Data and coordinate contracts are explicit.
- Experiments are reproducible and artifacts are traceable.
- Historical outputs no longer obscure the active path.

### Scientific success

- Claims correspond to controlled evidence.
- Failure modes are categorized and used to choose the next experiment.
- Simple baselines are retained and difficult to beat accidentally.
- Reward, optimization, and task success are not conflated.

### Robot-learning success

- A policy reliably completes the defined task in controlled evaluation.
- Performance generalizes across progressively broader, documented conditions.
- Physical trials are safe, reproducible, and measured.
- Any RL improvement is demonstrated against a strong supervised baseline.

The final ambition may extend beyond fixed-cube pickup to broader manipulation,
multiple objects, language-conditioned behavior, pretrained VLAs, or efficient
post-training. Those goals should be built on the trusted ladder rather than
used to bypass it.

---

## 28. Working rule for every next step

Before implementing or launching something substantial, ask:

1. What uncertainty are we reducing?
2. What is the simplest experiment that can reduce it?
3. What do we predict will happen?
4. What observation would prove us wrong?
5. Are the data, coordinates, and evaluation path already trusted at this layer?
6. Will the result remain interpretable if it is surprising?
7. Are we adding complexity because evidence requires it, or because it is
   available?

If those questions have clear answers, proceed. If they do not, the next step
is usually to inspect, simplify, or instrument the system—not to launch a
larger training run.

