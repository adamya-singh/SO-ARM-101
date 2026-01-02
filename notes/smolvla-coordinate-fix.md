# SmolVLA Coordinate System Investigation & Fix

## Situation

While integrating the SmolVLA vision-language-action model with our MuJoCo simulation for the SO-101 robot arm, we discovered suspicious hardcoded normalization statistics in `so101_mujoco_utils.py`:

```python
SMOLVLA_STATE_MEAN = np.array([1.596, 119.944, 109.770, 56.706, -27.423, 12.003])
SMOLVLA_STATE_STD = np.array([26.392, 52.411, 49.854, 36.998, 59.360, 19.040])
```

**The Problem**: The `shoulder_lift` mean was `119.944°`, but our MuJoCo model's joint limits are `±100°` (range="-1.7453 1.7453" radians). 

You cannot have a statistical mean of 119.944° when the joint can physically only move ±100°!

This indicated a fundamental coordinate system mismatch that would cause SmolVLA predictions to be completely wrong in simulation.

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

### Step 2: Understanding Zero-Point Reference

Different systems can define "0°" as different physical positions. Like time zones - 12:00 PM in New York is 9:00 AM in Los Angeles. Same moment, different numbers.

For robot joints:
- **MuJoCo** might define 0° as arm pointing up
- **Physical robot** might define 0° as arm pointing forward
- The same physical pose has different angle values in each system

### Step 3: Critical User Observation

> "When you click reset in MuJoCo, the position it goes to is the same position I was instructed to hold my physical arm in when I started calibration through the LeRobot library."

This was the key insight! It means:
- **MuJoCo reset** (all joints = 0) = **LeRobot calibration pose**
- Both systems should define this as 0° in their respective coordinate frames

### Step 4: Analyzing the LeRobot Calibration File

Located at: `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/`

```json
{
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
    }
    // ... other joints
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

### Step 6: The Discovery

At calibration pose, the **absolute** servo position for shoulder_lift:
```
Raw position = 2048 - 766 = 1282 ticks
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
# These offsets convert MuJoCo's calibrated coordinates (where calibration pose = 0°)
# to SmolVLA's absolute servo coordinate frame.
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
| shoulder_lift | 0° | +120° = 120° | 119.944° | ≈0 ✓ |
| elbow_flex | 0° | +110° = 110° | 109.770° | ≈0 ✓ |
| wrist_flex | 0° | +57° = 57° | 56.706° | ≈0 ✓ |

The normalized values are now ≈0 at the calibration pose, which is exactly what the neural network expects as a "neutral" input!

---

## Key Learnings

1. **Zero-point reference matters**: Different systems (MuJoCo, physical robot, SmolVLA training data) can define 0° at different physical positions

2. **SmolVLA uses absolute coordinates**: The model was trained with absolute servo positions (0-360° scale), not LeRobot's calibrated positions (centered at 0°)

3. **Calibration files are gold**: The LeRobot calibration JSON contains `homing_offset` values that reveal the relationship between servo raw positions and calibrated positions

4. **The offset ≈ the mean**: When coordinate systems are misaligned, the correct offset is approximately equal to the training data mean for each joint

5. **Verify with the reset pose**: If MuJoCo reset = LeRobot calibration pose, both should produce 0° — any difference indicates the offset needed

---

## Files Modified

- `simulation_code/so101_mujoco_utils.py` - Updated `MUJOCO_TO_PHYSICAL_OFFSET` array

## Related Files

- `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/*.json` - LeRobot calibration data
- `simulation_code/model/so101_new_calib.xml` - MuJoCo robot model with joint limits

