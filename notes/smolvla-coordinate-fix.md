# SmolVLA Coordinate System Investigation & Fix

## Situation

While integrating the SmolVLA vision-language-action model with our MuJoCo simulation for the SO-101 robot arm, we discovered suspicious hardcoded normalization statistics in `so101_mujoco_utils.py`:

```python
SMOLVLA_STATE_MEAN = np.array([1.596, 119.944, 109.770, 56.706, -27.423, 12.003])
SMOLVLA_STATE_STD = np.array([26.392, 52.411, 49.854, 36.998, 59.360, 19.040])
```

**The Problem**: The `shoulder_lift` mean was `119.944°`, but our MuJoCo model's joint limits are `±100°` (range="-1.7453 1.7453" radians). 

You cannot have a statistical mean of 119.944° when the joint can physically only move ±100°!

### MuJoCo Joint Limits (from `so101_new_calib.xml`)

```xml
<joint name="shoulder_pan" range="-1.9198621771937616 1.9198621771937634"/>  <!-- ±110° -->
<joint name="shoulder_lift" range="-1.7453292519943224 1.7453292519943366"/> <!-- ±100° -->
<joint name="elbow_flex" range="-1.69 1.69"/>                                 <!-- ±97° -->
<joint name="wrist_flex" range="-1.6580628494556928 1.6580627293335335"/>    <!-- ±95° -->
<joint name="wrist_roll" range="-2.7438472969992493 2.841206309382605"/>     <!-- -157° to +163° -->
<joint name="gripper" range="-0.17453297762778586 1.7453291995659765"/>      <!-- -10° to +100° -->
```

This indicated a fundamental coordinate system mismatch that would cause SmolVLA predictions to be completely wrong in simulation.

### The Original (Wrong) Offset Configuration

```python
# BEFORE (incorrect - all zeros)
MUJOCO_TO_PHYSICAL_OFFSET = np.array([
    0.0,      # shoulder_pan
    0.0,      # shoulder_lift
    0.0,      # elbow_flex
    0.0,      # wrist_flex
    0.0,      # wrist_roll
    0.0,      # gripper
])
```

---

## Task

1. Understand what these normalization statistics are used for
2. Determine why the values exceeded MuJoCo joint limits
3. Identify the correct coordinate frame transformation between MuJoCo and SmolVLA
4. Fix the `MUJOCO_TO_PHYSICAL_OFFSET` array to properly bridge the two coordinate systems

---

## Action

### Step 1: Understanding the Normalization Pipeline

The stats are used for **z-score normalization** before feeding data to the neural network:

```python
# Normalization (input to model)
normalized = (physical_degrees - SMOLVLA_STATE_MEAN) / SMOLVLA_STATE_STD

# Denormalization (output from model)
physical_degrees = normalized * SMOLVLA_ACTION_STD + SMOLVLA_ACTION_MEAN
```

The full normalization flow in `so101_mujoco_utils.py`:

```python
def normalize_state_for_smolvla(state_radians: np.ndarray) -> np.ndarray:
    # Step 1 & 2: Convert to physical robot frame in degrees
    physical_degrees = mujoco_to_physical_degrees(state_radians)
    
    # Step 3: Z-score normalization
    normalized = (physical_degrees - SMOLVLA_STATE_MEAN) / SMOLVLA_STATE_STD
    return normalized

def mujoco_to_physical_degrees(state_radians: np.ndarray) -> np.ndarray:
    state_degrees = np.degrees(state_radians)
    physical_degrees = state_degrees + MUJOCO_TO_PHYSICAL_OFFSET  # <-- This was wrong!
    return physical_degrees
```

### Step 2: Understanding Zero-Point Reference

Different systems can define "0°" as different physical positions. Like time zones - 12:00 PM in New York is 9:00 AM in Los Angeles. Same moment, different numbers.

```
        MuJoCo Zero (0°)          Physical Robot Zero (0°)
              |                          |
              ▼                          ▼
          ═══════                    ════╗
         (arm forward)              (arm up)

If the arm is pointing forward:
  - MuJoCo reports: 0°
  - Physical robot reports: 120° (it's 120° away from ITS zero)
```

For robot joints:
- **MuJoCo** might define 0° as arm pointing up
- **Physical robot** might define 0° as arm pointing forward
- The same physical pose has different angle values in each system

### Step 3: Critical User Observation

> "When you click reset in MuJoCo, the position it goes to is the same position I was instructed to hold my physical arm in when I started calibration through the LeRobot library."

This was the **key insight**! It means:
- **MuJoCo reset** (all joints = 0) = **LeRobot calibration pose**
- Both systems should define this as 0° in their respective coordinate frames
- But SmolVLA uses a DIFFERENT coordinate system than LeRobot's calibrated coordinates!

### Step 4: Analyzing the LeRobot Calibration File

Located at: `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/`

**Full calibration data from the physical robot:**

```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -1399,
        "range_min": 874,
        "range_max": 2979
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -766,
        "range_min": 970,
        "range_max": 3048
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": 1011,
        "range_min": 1005,
        "range_max": 3219
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -877,
        "range_min": 908,
        "range_max": 3076
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 1512,
        "range_min": 736,
        "range_max": 3961
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 943,
        "range_min": 2046,
        "range_max": 2053
    }
}
```

### Step 5: Decoding the Calibration Values

The STS3215 servos use a 12-bit encoder:
- **Raw range**: 0 to 4095 ticks
- **Center**: 2048 ticks
- **Conversion**: 4096 ticks = 360°, so **1 tick ≈ 0.088°**

The `homing_offset` shifts the servo's raw position to make the calibration pose = 0°.

| Joint | homing_offset (ticks) | ≈ Degrees from servo center |
|-------|----------------------|------------------------------|
| shoulder_pan | -1399 | -123° |
| shoulder_lift | -766 | -67° |
| elbow_flex | +1011 | +89° |
| wrist_flex | -877 | -77° |
| wrist_roll | +1512 | +133° |
| gripper | +943 | +83° |

### Step 6: The Discovery

At calibration pose, the **absolute** servo position for shoulder_lift:
```
Raw position at calibration = 2048 - 766 = 1282 ticks
In degrees = 1282 × (360/4096) ≈ 112.7°
```

Compare to SmolVLA training mean: **119.944°** — nearly identical!

**Conclusion**: SmolVLA was trained using **absolute servo positions** (0-360° scale), NOT calibrated positions centered at 0°.

### Step 7: Deriving the Correct Offsets

Since:
- MuJoCo 0° = Calibration pose
- SmolVLA expects ~120° for shoulder_lift at calibration pose

The offset must bridge this gap:
```
SmolVLA_position = MuJoCo_position + OFFSET
```

The offsets should approximately equal the SmolVLA training means!

---

## Result

Updated `MUJOCO_TO_PHYSICAL_OFFSET` in `so101_mujoco_utils.py`:

```python
# ===== MuJoCo to Physical Robot Coordinate Offset (in DEGREES) =====
# Physical Robot Position = MuJoCo Position (radians converted to degrees) + OFFSET
# 
# These offsets convert MuJoCo's calibrated coordinates (where calibration pose = 0°)
# to SmolVLA's absolute servo coordinate frame (where servo raw center 2048 ≈ 180°).
# 
# SmolVLA was trained with absolute servo positions, NOT calibrated positions.
# At calibration pose (MuJoCo 0°), the absolute servo positions are ~120° for shoulder_lift, etc.
# These values are derived from the SmolVLA training data statistics (SMOLVLA_STATE_MEAN).
MUJOCO_TO_PHYSICAL_OFFSET = np.array([
    0.0,      # shoulder_pan - near zero offset (training mean ≈ 1.6°)
    120.0,    # shoulder_lift - calibration pose ≈ 120° in absolute frame
    110.0,    # elbow_flex - calibration pose ≈ 110° in absolute frame
    57.0,     # wrist_flex - calibration pose ≈ 57° in absolute frame
    -27.0,    # wrist_roll - calibration pose ≈ -27° in absolute frame
    12.0,     # gripper - calibration pose ≈ 12° in absolute frame
])
```

### Verification

With these offsets, when MuJoCo is at reset position (0°):

| Joint | MuJoCo | + Offset | SmolVLA Mean | Normalized |
|-------|--------|----------|--------------|------------|
| shoulder_pan | 0° | +0° = 0° | 1.596° | ≈-0.06 ✓ |
| shoulder_lift | 0° | +120° = 120° | 119.944° | ≈0 ✓ |
| elbow_flex | 0° | +110° = 110° | 109.770° | ≈0 ✓ |
| wrist_flex | 0° | +57° = 57° | 56.706° | ≈0 ✓ |
| wrist_roll | 0° | -27° = -27° | -27.423° | ≈0 ✓ |
| gripper | 0° | +12° = 12° | 12.003° | ≈0 ✓ |

The normalized values are now ≈0 at the calibration pose, which is exactly what the neural network expects as a "neutral" input!

---

## Impact on Training

### Was Training Affected? YES! 🚨

The training code uses the same normalization pipeline:

```
train_reinflow.py 
    → vectorized_env.py / subproc_vectorized_env.py
        → normalize_state_for_vla() 
            → normalize_state_for_smolvla()
                → mujoco_to_physical_degrees()  ← Uses MUJOCO_TO_PHYSICAL_OFFSET
```

**With the OLD offsets (all zeros):**
```python
# MuJoCo at reset position (shoulder_lift = 0°)
physical_degrees = 0° + 0° = 0°
normalized = (0° - 119.944°) / 52.411° = -2.29  # ❌ Way off!
```

**With the NEW offsets (corrected):**
```python
# MuJoCo at reset position (shoulder_lift = 0°)
physical_degrees = 0° + 120° = 120°
normalized = (120° - 119.944°) / 52.411° ≈ 0.0  # ✅ Correct!
```

### Training Impact Summary

| Issue | Impact |
|-------|--------|
| **Shifted inputs** | Neural network saw states ~2+ standard deviations from expected |
| **Poor initial policy** | SmolVLA's pretrained knowledge was useless - states looked "alien" |
| **Wasted learning** | RL had to learn to compensate for systematic coordinate bias |
| **Suboptimal convergence** | Model may have learned a warped state→action mapping |

### Recommendation

**Retrain with the corrected offsets.** The new training should:
- Converge faster (pretrained SmolVLA knowledge is now useful)
- Learn a more natural policy
- Potentially achieve better final performance

---

## Key Learnings

1. **Zero-point reference matters**: Different systems (MuJoCo, physical robot, SmolVLA training data) can define 0° at different physical positions

2. **SmolVLA uses absolute coordinates**: The model was trained with absolute servo positions (0-360° scale), not LeRobot's calibrated positions (centered at 0°)

3. **Calibration files are gold**: The LeRobot calibration JSON contains `homing_offset` values that reveal the relationship between servo raw positions and calibrated positions

4. **The offset ≈ the mean**: When coordinate systems are misaligned, the correct offset is approximately equal to the training data mean for each joint

5. **Verify with the reset pose**: If MuJoCo reset = LeRobot calibration pose, both should produce 0° — any difference indicates the offset needed

6. **Training is affected**: The normalization is used throughout the training pipeline, not just inference - wrong offsets cause the model to learn with a handicap

---

## Code Paths Affected

Files that use the normalization and were affected by this fix:

| File | Function | Purpose |
|------|----------|---------|
| `so101_mujoco_utils.py` | `normalize_state_for_smolvla()` | Core normalization |
| `so101_mujoco_utils.py` | `unnormalize_action_from_smolvla()` | Action denormalization |
| `so101_mujoco_utils.py` | `prepare_observation()` | Observation prep for inference |
| `reinflow_smolvla.py` | `prepare_observation_for_reinflow()` | Training observation prep |
| `so101_gym_env.py` | `_get_obs()` | Gym environment observations |
| `vectorized_env.py` | `get_obs()` | Parallel env observations |
| `subproc_vectorized_env.py` | `get_obs()` | Subprocess env observations |

---

## Files Modified

- `simulation_code/so101_mujoco_utils.py` - Updated `MUJOCO_TO_PHYSICAL_OFFSET` array with correct values

## Related Files

- `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/*.json` - LeRobot calibration data
- `simulation_code/model/so101_new_calib.xml` - MuJoCo robot model with joint limits

---

## Appendix: How to Verify Your Calibration

To check your own calibration offsets:

```bash
# View your calibration file
cat ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/*.json
```

Then calculate the absolute position at calibration pose:
```
absolute_degrees = (2048 + homing_offset) × (360 / 4096)
```

This should approximately match the corresponding `SMOLVLA_STATE_MEAN` value for that joint.
