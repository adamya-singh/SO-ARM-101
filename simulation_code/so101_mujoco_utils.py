import time
import warnings
import mujoco
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Any

# ===== MuJoCo to Physical Robot Coordinate Offset (in DEGREES) =====
# Physical Robot Position = MuJoCo Position (radians converted to degrees) + OFFSET
# 
# These offsets convert MuJoCo's calibrated coordinates (where calibration pose = 0°)
# to SmolVLA's absolute servo coordinate frame (where servo raw center 2048 ≈ 180°).
# 
# These constants are only for the legacy/base SmolVLA path that used an
# absolute-servo frame with hardcoded z-score stats.
# Finetuned SmolVLA processor-backed checkpoints instead use calibrated radians.
MUJOCO_TO_PHYSICAL_OFFSET = np.array([
    0.0,      # shoulder_pan - near zero offset (training mean ≈ 1.6°)
    120.0,    # shoulder_lift - calibration pose ≈ 120° in absolute frame
    110.0,    # elbow_flex - calibration pose ≈ 110° in absolute frame
    57.0,     # wrist_flex - calibration pose ≈ 57° in absolute frame
    -27.0,    # wrist_roll - calibration pose ≈ -27° in absolute frame
    12.0,     # gripper - calibration pose ≈ 12° in absolute frame
])

# ===== SmolVLA Normalization Stats (in DEGREES) =====
# These are the mean/std values from the SO-100 training data
# Used for z-score normalization: normalized = (value - mean) / std
SMOLVLA_STATE_MEAN = np.array([1.596, 119.944, 109.770, 56.706, -27.423, 12.003])
SMOLVLA_STATE_STD = np.array([26.392, 52.411, 49.854, 36.998, 59.360, 19.040])
SMOLVLA_ACTION_MEAN = np.array([1.596, 119.944, 109.770, 56.706, -27.423, 12.003])
SMOLVLA_ACTION_STD = np.array([26.392, 52.411, 49.854, 36.998, 59.360, 19.040])

# ===== Pi0 Processor Cache =====
_cached_pi0_preprocessor = None
_cached_pi0_postprocessor = None
_cached_pi0_pretrained_path = None
_cached_smolvla_preprocessor = None
_cached_smolvla_postprocessor = None
_cached_smolvla_pretrained_path = None


def resolve_policy_artifact_path(pretrained_path: str) -> Tuple[str, str]:
    """
    Resolve a policy source into a concrete local artifact path when applicable.

    Returns:
        (resolved_path, source_type) where source_type is "hf_repo" or "local_checkpoint".
    """
    path = Path(pretrained_path).expanduser()
    if not path.exists():
        return pretrained_path, "hf_repo"

    if not path.is_dir():
        raise FileNotFoundError(
            f"Local policy path is not a directory: {pretrained_path}\n"
            f"Expected a Hugging Face repo id or a local LeRobot model/checkpoint directory."
        )

    pretrained_model_dir = path / "pretrained_model"
    if pretrained_model_dir.is_dir():
        return str(pretrained_model_dir), "local_checkpoint"
    if (path / "config.json").exists():
        return str(path), "local_checkpoint"

    raise FileNotFoundError(
        f"Local policy path exists but is not a valid LeRobot model artifact: {pretrained_path}\n"
        f"Expected either:\n"
        f"  - a model directory containing config.json, or\n"
        f"  - a checkpoint directory containing pretrained_model/"
    )


def detect_smolvla_image_keys(policy) -> list[str]:
    """Return the image keys expected by the loaded SmolVLA policy."""
    image_features = getattr(policy.config, "image_features", None)
    if image_features:
        return list(image_features.keys())

    input_features = getattr(policy.config, "input_features", {})
    image_keys = [key for key in input_features if key.startswith("observation.images.")]
    if image_keys:
        return image_keys

    raise ValueError("Could not determine SmolVLA image feature keys from policy config.")


def build_smolvla_image_observation(
    expected_image_keys: list[str],
    image_top_tensor: torch.Tensor,
    image_wrist_tensor: torch.Tensor,
    image_side_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map MuJoCo camera tensors onto the image keys expected by the SmolVLA policy."""
    alias_map = {
        "observation.images.camera1": image_top_tensor,
        "observation.images.top": image_top_tensor,
        "observation.images.camera_up": image_top_tensor,
        "observation.images.overhead": image_top_tensor,
        "observation.images.camera2": image_wrist_tensor,
        "observation.images.wrist": image_wrist_tensor,
        "observation.images.camera3": image_side_tensor,
        "observation.images.side": image_side_tensor,
    }

    observation = {}
    missing_keys = []
    for key in expected_image_keys:
        if key in alias_map:
            observation[key] = alias_map[key]
        else:
            missing_keys.append(key)

    if missing_keys:
        raise KeyError(
            "Unsupported SmolVLA image feature keys for MuJoCo observation mapping: "
            f"{missing_keys}. Supported keys: {sorted(alias_map.keys())}"
        )

    return observation


def _extract_processed_state(processed: Any, fallback: np.ndarray) -> np.ndarray:
    """Extract observation.state from a processor output, falling back if unavailable."""
    if isinstance(processed, dict) and "observation" in processed:
        state = processed["observation"].get("observation.state")
        if isinstance(state, torch.Tensor):
            return state.squeeze(0).detach().cpu().numpy()
    if isinstance(processed, dict) and "observation.state" in processed:
        state = processed["observation.state"]
        if isinstance(state, torch.Tensor):
            return state.squeeze(0).detach().cpu().numpy()
    return fallback


def load_pi0_processors(pretrained_path: str = "lerobot/pi0", policy_config=None) -> Tuple[Any, Any]:
    """
    Load preprocessor and postprocessor for Pi0 model.
    
    Pi0 uses similar normalization to SmolVLA, so we use the same
    PolicyProcessorPipeline approach.
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        policy_config: Optional PI0Config for creating default processors
        
    Returns:
        (preprocessor, postprocessor) tuple for normalizing inputs and denormalizing outputs
    """
    global _cached_pi0_preprocessor, _cached_pi0_postprocessor, _cached_pi0_pretrained_path
    
    # Return cached processors if already loaded for this path
    if _cached_pi0_pretrained_path == pretrained_path and _cached_pi0_preprocessor is not None:
        return _cached_pi0_preprocessor, _cached_pi0_postprocessor
    
    try:
        from lerobot.processor import PolicyProcessorPipeline
    except ImportError as e:
        raise ImportError(
            f"Could not import PolicyProcessorPipeline from LeRobot.\n"
            f"Original error: {e}"
        ) from e
    
    print(f"[Pi0] Loading processors from {pretrained_path}...")
    
    preprocessor = None
    postprocessor = None
    hub_load_failed = False
    
    # Try to load from HuggingFace Hub first
    try:
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="preprocessor_config.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="postprocessor_config.json"
        )
        print(f"[Pi0] Processors loaded from Hub successfully!")
    except (FileNotFoundError, Exception) as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg or "Could not find" in error_msg:
            print(f"[Pi0] Processor configs not found on Hub (this is normal for older models)")
            hub_load_failed = True
        else:
            raise RuntimeError(
                f"Failed to load processors from '{pretrained_path}'.\n"
                f"Original error: {e}"
            ) from e
    
    # Fall back to creating default processors from policy config
    if hub_load_failed:
        if policy_config is None:
            print(f"[Pi0] Warning: No processors available, using identity normalization")
            _cached_pi0_preprocessor = None
            _cached_pi0_postprocessor = None
            _cached_pi0_pretrained_path = pretrained_path
            return None, None
        
        try:
            # Try to create default processors using Pi0's processor
            from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
            print(f"[Pi0] Creating default processors from policy config (identity normalization)...")
            preprocessor, postprocessor = make_pi0_pre_post_processors(
                config=policy_config,
                dataset_stats=None,
            )
            print(f"[Pi0] Default processors created successfully!")
        except Exception as e:
            print(f"[Pi0] Warning: Could not create Pi0 processors: {e}")
            print(f"[Pi0] Using identity normalization")
            preprocessor = None
            postprocessor = None
    
    # Cache for future use
    _cached_pi0_preprocessor = preprocessor
    _cached_pi0_postprocessor = postprocessor
    _cached_pi0_pretrained_path = pretrained_path
    
    return preprocessor, postprocessor


def load_smolvla_processors(pretrained_path: str = "lerobot/smolvla_base", policy_config=None) -> Tuple[Any, Any]:
    """
    Load preprocessor and postprocessor for SmolVLA model artifacts.

    If processor configs are missing, return (None, None) and let callers use
    the legacy hardcoded SmolVLA z-score stats.
    """
    del policy_config  # SmolVLA processor loading is artifact-driven.
    global _cached_smolvla_preprocessor, _cached_smolvla_postprocessor, _cached_smolvla_pretrained_path

    if _cached_smolvla_pretrained_path == pretrained_path and _cached_smolvla_preprocessor is not None:
        return _cached_smolvla_preprocessor, _cached_smolvla_postprocessor

    try:
        from lerobot.processor import PolicyProcessorPipeline
    except ImportError as e:
        raise ImportError(
            f"Could not import PolicyProcessorPipeline from LeRobot.\n"
            f"Original error: {e}"
        ) from e

    print(f"[SmolVLA] Attempting to load processors from {pretrained_path}...")

    preprocessor = None
    postprocessor = None
    try:
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="policy_preprocessor.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="policy_postprocessor.json"
        )
        print("[SmolVLA] Loaded processor-backed normalization from policy artifact")
    except (FileNotFoundError, Exception) as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg or "Could not find" in error_msg:
            print("[SmolVLA] Processor configs not found; using legacy hardcoded normalization fallback")
        else:
            print(f"[SmolVLA] Processor loading failed; using legacy hardcoded normalization fallback: {e}")
        preprocessor = None
        postprocessor = None

    _cached_smolvla_preprocessor = preprocessor
    _cached_smolvla_postprocessor = postprocessor
    _cached_smolvla_pretrained_path = pretrained_path
    return preprocessor, postprocessor


def load_vla_processors(model_type: str, pretrained_path: str, policy_config=None) -> Tuple[Any, Any]:
    """
    Load processors for either SmolVLA or Pi0 model.
    
    This is the unified dispatcher function that selects the correct
    processor loading function based on model type.
    
    Args:
        model_type: "smolvla" or "pi0"
        pretrained_path: HuggingFace model path or local checkpoint
        policy_config: Optional policy config for creating default processors
        
    Returns:
        (preprocessor, postprocessor) tuple. SmolVLA returns (None, None) only
        when processor configs are unavailable and legacy hardcoded stats should be used.
    """
    if model_type == "smolvla":
        return load_smolvla_processors(pretrained_path, policy_config)
    elif model_type == "pi0":
        return load_pi0_processors(pretrained_path, policy_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'smolvla' or 'pi0'.")


def normalize_state_for_vla(
    state_radians: np.ndarray,
    model_type: str,
    preprocessor=None,
    instruction: Optional[str] = None,
) -> np.ndarray:
    """
    Model-agnostic state normalization.
    
    Routes to the correct normalization function based on model type.
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
        model_type: "smolvla" or "pi0"
        preprocessor: PolicyProcessorPipeline for model-backed normalization
        instruction: Task instruction required by processor-backed SmolVLA normalization
        
    Returns:
        normalized state as numpy array (6,)
    """
    if model_type == "smolvla":
        return normalize_state_for_smolvla(
            state_radians,
            preprocessor=preprocessor,
            instruction=instruction,
        )
    elif model_type == "pi0":
        return normalize_state_for_pi0(state_radians, preprocessor)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'smolvla' or 'pi0'.")


def unnormalize_action_for_vla(action_normalized: np.ndarray, model_type: str, postprocessor=None) -> np.ndarray:
    """
    Model-agnostic action denormalization.
    
    Routes to the correct denormalization function based on model type.
    
    Args:
        action_normalized: numpy array of normalized actions
        model_type: "smolvla" or "pi0"
        postprocessor: PolicyProcessorPipeline for model-backed denormalization
        
    Returns:
        action in MuJoCo radians as numpy array
    """
    if model_type == "smolvla":
        return unnormalize_action_from_smolvla(action_normalized, postprocessor=postprocessor)
    elif model_type == "pi0":
        return unnormalize_action_for_pi0(action_normalized, postprocessor)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'smolvla' or 'pi0'.")


def mujoco_to_physical_degrees(state_radians: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo joint state (radians) to physical robot frame in DEGREES.
    
    Pipeline: radians -> degrees -> add offset
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
        
    Returns:
        state in physical robot frame in degrees
    """
    state_degrees = np.degrees(state_radians)
    physical_degrees = state_degrees + MUJOCO_TO_PHYSICAL_OFFSET
    return physical_degrees


def physical_degrees_to_mujoco(physical_degrees: np.ndarray) -> np.ndarray:
    """
    Convert physical robot frame (degrees) to MuJoCo frame (radians).
    
    Pipeline: subtract offset -> degrees -> radians
    
    Args:
        physical_degrees: numpy array of actions in physical robot frame in degrees (6,)
        
    Returns:
        action in MuJoCo radians
    """
    state_degrees = physical_degrees - MUJOCO_TO_PHYSICAL_OFFSET
    state_radians = np.radians(state_degrees)
    return state_radians


def mujoco_to_physical_state(state_radians: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo joint state (radians) to physical robot frame.
    
    This handles coordinate frame conversion WITHOUT normalization.
    The physical frame is what the VLA model expects as input.
    
    Note: For Pi0 processor-based approach, this returns radians with offset.
    For SmolVLA hardcoded approach, use mujoco_to_physical_degrees() instead.
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
        
    Returns:
        state in physical robot frame (radians with offset applied)
    """
    # Apply offset (converted to radians) for Pi0 processor-based approach
    physical_state = state_radians + np.radians(MUJOCO_TO_PHYSICAL_OFFSET)
    return physical_state


def physical_to_mujoco_action(physical_action: np.ndarray) -> np.ndarray:
    """
    Convert action from physical robot frame to MuJoCo frame (radians).
    
    This handles coordinate frame conversion WITHOUT normalization.
    
    Note: For Pi0 processor-based approach only.
    For SmolVLA hardcoded approach, use physical_degrees_to_mujoco() instead.
    
    Args:
        physical_action: numpy array of actions in physical robot frame (radians) (6,)
        
    Returns:
        action in MuJoCo radians
    """
    # Remove offset (converted to radians) for Pi0 processor-based approach
    mujoco_action = physical_action - np.radians(MUJOCO_TO_PHYSICAL_OFFSET)
    return mujoco_action

def convert_to_dictionary(qpos):
    return {
            'shoulder_pan': qpos[0]*180.0/3.14159,  # convert radians(mujoco) to degrees(SO101)
            'shoulder_lift': qpos[1]*180.0/3.14159,
            'elbow_flex': qpos[2]*180.0/3.14159,
            'wrist_flex': qpos[3]*180.0/3.14159,
            'wrist_roll': qpos[4]*180.0/3.14159,
            'gripper': qpos[5]*100/3.14159,
        }

def convert_to_list(dictionary):
    return [
            dictionary['shoulder_pan']*3.14159/180.0,   # convert degrees(SO101) to radians(mujoco)
            dictionary['shoulder_lift']*3.14159/180.0,
            dictionary['elbow_flex']*3.14159/180.0,
            dictionary['wrist_flex']*3.14159/180.0,
            dictionary['wrist_roll']*3.14159/180.0,
            dictionary['gripper']*3.14159/100.0,
        ]

def set_initial_pose(d, position_dict):
    pos = convert_to_list(position_dict)
    d.qpos[:6] = pos  # Only set the first 6 elements (robot joints)

def send_position_command(d, position_dict):
    pos = convert_to_list(position_dict)
    d.ctrl = pos

def move_to_pose(m, d, viewer, desired_position, duration):
    start_time = time.time()
    starting_pose = d.qpos[:6].copy()  # Only get robot joints
    starting_pose = convert_to_dictionary(starting_pose)

    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # interpolation factor [0,1] (make sure it doesn't exceed 1)
        alpha = min(t / duration, 1)

        # interpolate each joint
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1 - alpha) * p0 + alpha * pf
        
        # send command to sim
        send_position_command(d, position_dict)
        mujoco.mj_step(m, d)

        # pick up changes to the physics state, apply peturbations, update options from GUI
        viewer.sync()

def hold_position(m, d, viewer, duration):
    current_pos = d.qpos[:6].copy()  # Only get robot joints
    current_pos_dict = convert_to_dictionary(current_pos)

    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        send_position_command(d, current_pos_dict)
        mujoco.mj_step(m, d)
        viewer.sync()

# ===== SmolVLA Helper Functions =====

def _legacy_normalize_state_for_smolvla(state_radians: np.ndarray) -> np.ndarray:
    """Legacy SmolVLA normalization in the servo-frame fallback path."""
    physical_degrees = mujoco_to_physical_degrees(state_radians)
    return (physical_degrees - SMOLVLA_STATE_MEAN) / SMOLVLA_STATE_STD


def _process_action_with_postprocessor(action: np.ndarray | torch.Tensor, postprocessor, model_label: str) -> np.ndarray:
    """Run a policy postprocessor through its action-specific API when available."""
    if torch.is_tensor(action):
        action_tensor = action.detach().to(dtype=torch.float32)
    else:
        action_tensor = torch.as_tensor(action, dtype=torch.float32)

    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    elif action_tensor.ndim != 2:
        raise RuntimeError(
            f"{model_label} postprocessor expected action rank 1 or 2, got shape {tuple(action_tensor.shape)}"
        )

    if hasattr(postprocessor, "process_action"):
        processed = postprocessor.process_action(action_tensor)
    else:
        processed = postprocessor(action_tensor)

    if torch.is_tensor(processed):
        processed_np = processed.detach().cpu().numpy()
    else:
        processed_np = np.asarray(processed)

    if processed_np.ndim == 2:
        if processed_np.shape[0] != 1:
            raise RuntimeError(
                f"{model_label} postprocessor returned batched shape {processed_np.shape}, expected batch size 1"
            )
        processed_np = processed_np[0]
    elif processed_np.ndim != 1:
        raise RuntimeError(
            f"{model_label} postprocessor returned invalid shape {processed_np.shape}, expected (action_dim,) or (1, action_dim)"
        )

    return processed_np.astype(np.float32, copy=False)


def normalize_state_for_smolvla(
    state_radians: np.ndarray,
    preprocessor=None,
    instruction: Optional[str] = None,
) -> np.ndarray:
    """
    Normalize robot state for SmolVLA input.
    
    Pipeline:
    1. If processor-backed: feed calibrated MuJoCo radians directly to the processor.
    2. Otherwise: convert MuJoCo calibrated frame -> legacy servo frame and apply
       the hardcoded SmolVLA z-score stats.
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
    
    Returns:
        normalized state as numpy array (6,)
    """
    if preprocessor is None:
        return _legacy_normalize_state_for_smolvla(state_radians)

    if not instruction:
        raise ValueError(
            "Processor-backed SmolVLA normalization requires a non-empty task instruction."
        )

    try:
        calibrated_state = state_radians.astype(np.float32, copy=False)
        obs_dict = {
            "observation.state": torch.from_numpy(calibrated_state).float().unsqueeze(0),
            "task": instruction,
        }
        processed = preprocessor(obs_dict)
        return _extract_processed_state(processed, calibrated_state)
    except Exception as e:
        warnings.warn(f"SmolVLA preprocessor failed, using legacy hardcoded normalization: {e}")
        return _legacy_normalize_state_for_smolvla(state_radians)


def normalize_state_for_pi0(state_radians: np.ndarray, preprocessor) -> np.ndarray:
    """
    Normalize robot state for Pi0 input using processor pipeline.
    
    Pipeline:
    1. MuJoCo radians -> physical robot frame (apply offset in radians)
    2. Apply preprocessor normalization (if available)
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
        preprocessor: PolicyProcessorPipeline for normalization (can be None for identity)
    
    Returns:
        normalized state as numpy array (6,)
    """
    # Step 1: Convert to physical robot frame (radians)
    physical_state = mujoco_to_physical_state(state_radians)
    
    # Step 2: Apply preprocessor normalization if available
    if preprocessor is None:
        # Identity normalization - just return physical state
        return physical_state
    
    try:
        obs_dict = {"observation.state": torch.from_numpy(physical_state).float().unsqueeze(0)}
        processed = preprocessor(obs_dict)
        
        # Handle different preprocessor return formats
        if isinstance(processed, dict) and "observation" in processed:
            normalized = processed["observation"]["observation.state"].squeeze(0).numpy()
        elif "observation.state" in processed:
            normalized = processed["observation.state"].squeeze(0).numpy()
        else:
            # Fallback: return physical state if we can't find the expected key
            return physical_state
        
        return normalized
    except Exception as e:
        # On error, fall back to identity normalization with warning
        import warnings
        warnings.warn(f"Pi0 preprocessor failed, using identity normalization: {e}")
        return physical_state


def unnormalize_action_from_smolvla(action_normalized: np.ndarray, postprocessor=None) -> np.ndarray:
    """
    Unnormalize action output from SmolVLA.
    
    Pipeline:
    1. If processor-backed: treat postprocessor output as calibrated MuJoCo radians.
    2. Otherwise: use legacy hardcoded denormalization in the servo frame and
       convert back to MuJoCo calibrated radians.
    
    Args:
        action_normalized: numpy array of normalized actions (6,) or (chunk_size, 6)
    
    Returns:
        action in MuJoCo radians as numpy array
    """
    # Handle both single action and action chunks
    original_shape = action_normalized.shape
    is_chunk = len(original_shape) > 1
    
    if is_chunk:
        # Process each action in the chunk
        results = []
        for action in action_normalized:
            result = _unnormalize_single_action_smolvla(action, postprocessor)
            results.append(result)
        return np.stack(results)
    else:
        return _unnormalize_single_action_smolvla(action_normalized, postprocessor)


def _unnormalize_single_action_smolvla(action_normalized: np.ndarray, postprocessor=None) -> np.ndarray:
    """Helper to unnormalize a single SmolVLA action."""
    if postprocessor is None:
        physical_degrees = action_normalized * SMOLVLA_ACTION_STD + SMOLVLA_ACTION_MEAN
        return physical_degrees_to_mujoco(physical_degrees)

    try:
        calibrated_action = _process_action_with_postprocessor(
            action_normalized,
            postprocessor,
            model_label="SmolVLA",
        )
    except Exception as e:
        raise RuntimeError(
            f"SmolVLA postprocessor failed to denormalize action.\n"
            f"Input action: {action_normalized}\n"
            f"Original error: {e}"
        ) from e

    return calibrated_action


def unnormalize_action_for_pi0(action_normalized: np.ndarray, postprocessor) -> np.ndarray:
    """
    Unnormalize action output from Pi0 using processor pipeline.
    
    Pipeline:
    1. Apply postprocessor denormalization (if available)
    2. Physical robot frame -> MuJoCo radians (remove offset)
    
    Args:
        action_normalized: numpy array of normalized actions (6,) or (chunk_size, 6)
        postprocessor: PolicyProcessorPipeline for denormalization (can be None for identity)
    
    Returns:
        action in MuJoCo radians as numpy array
    """
    if postprocessor is None:
        # Identity denormalization - just convert frame
        return physical_to_mujoco_action(action_normalized)
    
    # Handle both single action and action chunks
    original_shape = action_normalized.shape
    is_chunk = len(original_shape) > 1
    
    if is_chunk:
        # Process each action in the chunk
        results = []
        for action in action_normalized:
            result = _unnormalize_single_action_pi0(action, postprocessor)
            results.append(result)
        return np.stack(results)
    else:
        return _unnormalize_single_action_pi0(action_normalized, postprocessor)


def _unnormalize_single_action_pi0(action_normalized: np.ndarray, postprocessor) -> np.ndarray:
    """Helper to unnormalize a single Pi0 action using processor."""
    # Apply postprocessor denormalization
    try:
        physical_action = _process_action_with_postprocessor(
            action_normalized,
            postprocessor,
            model_label="Pi0",
        )
    except Exception as e:
        raise RuntimeError(
            f"Pi0 postprocessor failed to denormalize action.\n"
            f"Input action: {action_normalized}\n"
            f"Original error: {e}"
        ) from e
    
    # Convert from physical robot frame to MuJoCo radians
    mujoco_action = physical_to_mujoco_action(physical_action)
    return mujoco_action


def get_camera_observation(renderer, d, camera_name="wrist_camera"):
    """Render a camera and return RGB image as numpy array."""
    renderer.update_scene(d, camera=camera_name)
    rgb_array = renderer.render()
    return rgb_array

def get_robot_state(d):
    """Get current joint positions only."""
    # SmolVLA base expects 6-dim state (joint positions only, no velocities)
    state = d.qpos[:6].copy()
    return state

def prepare_observation(rgb_image_top, rgb_image_wrist, rgb_image_side, robot_state, instruction, device, policy=None, preprocessor=None, model_type="smolvla", debug=False):
    """
    Prepare observation dict for VLA policy (SmolVLA or Pi0) with multiple cameras.
    
    For SmolVLA: Uses calibrated-radians processor normalization when available,
    otherwise the legacy servo-frame hardcoded stats.
    For Pi0: Uses processor pipeline for normalization.
    
    Args:
        rgb_image_top: numpy array of shape (H, W, C) from top camera with values in [0, 255]
        rgb_image_wrist: numpy array of shape (H, W, C) from wrist camera with values in [0, 255]
        rgb_image_side: numpy array of shape (H, W, C) from side camera with values in [0, 255]
        robot_state: numpy array of robot state in RADIANS (from MuJoCo)
        instruction: string with task instruction
        device: torch device (cuda, mps, or cpu)
        policy: VLA policy object (used for tokenization)
        preprocessor: PolicyProcessorPipeline for model-backed preprocessing
        model_type: "smolvla" or "pi0" - determines normalization approach
    
    Returns:
        observation: dict with images, state, and language tensors using VLA standard keys
    """
    # Step 1: Convert images to tensors with [0, 1] range and (C, H, W) format
    def numpy_to_image_tensor(img_numpy):
        """Convert (H, W, C) uint8 numpy to (C, H, W) float tensor in [0, 1]."""
        img_tensor = torch.from_numpy(img_numpy).float() / 255.0
        return img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    image_top_tensor = numpy_to_image_tensor(rgb_image_top)
    image_wrist_tensor = numpy_to_image_tensor(rgb_image_wrist)
    image_side_tensor = numpy_to_image_tensor(rgb_image_side)
    
    # Step 2: Normalize state based on model type
    if model_type == "smolvla":
        normalized_state = normalize_state_for_smolvla(
            robot_state,
            preprocessor=preprocessor,
            instruction=instruction,
        )
        state_tensor = torch.from_numpy(normalized_state).float()
    elif model_type == "pi0" and preprocessor is not None:
        # Pi0 with preprocessor - use processor pipeline path
        physical_state = mujoco_to_physical_state(robot_state)
        state_tensor = torch.from_numpy(physical_state).float()
        
        # Build batch for preprocessor
        batch = {
            "observation.images.camera1": image_top_tensor,
            "observation.images.camera2": image_wrist_tensor,
            "observation.images.camera3": image_side_tensor,
            "observation.state": state_tensor,
            "task": instruction,
        }
        
        try:
            processed = preprocessor(batch)
            observation = {}
            for key, value in processed.items():
                if isinstance(value, torch.Tensor):
                    observation[key] = value.to(device)
                else:
                    observation[key] = value
            
            if debug:
                print(f"\n[Observation Preparation Debug - Pi0 Preprocessor Path]")
                print(f"  Instruction: '{instruction}'")
                for key, val in observation.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            
            return observation
        except Exception as e:
            print(f"[Warning] Pi0 preprocessor failed: {e}")
            print(f"[Warning] Falling back to identity normalization")
            # Fall through to manual path with identity normalization
            physical_state = mujoco_to_physical_state(robot_state)
            state_tensor = torch.from_numpy(physical_state).float()
    else:
        # Pi0 without preprocessor - identity normalization
        physical_state = mujoco_to_physical_state(robot_state)
        state_tensor = torch.from_numpy(physical_state).float()
    
    # Step 3: Manual observation building (SmolVLA path, or Pi0 fallback)
    # Add batch dimension to images and state
    image_top_tensor = image_top_tensor.unsqueeze(0).to(device)
    image_wrist_tensor = image_wrist_tensor.unsqueeze(0).to(device)
    image_side_tensor = image_side_tensor.unsqueeze(0).to(device)
    state_tensor = state_tensor.unsqueeze(0).to(device)
    
    # Tokenization
    if policy is not None and hasattr(policy, 'tokenizer'):
        tokens = policy.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        language_tokens = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].bool().to(device)
    else:
        # Dummy tokens as last resort
        language_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
        attention_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
    
    if model_type == "smolvla" and policy is not None:
        expected_image_keys = detect_smolvla_image_keys(policy)
        image_observation = build_smolvla_image_observation(
            expected_image_keys,
            image_top_tensor,
            image_wrist_tensor,
            image_side_tensor,
        )
    else:
        image_observation = {
            "observation.images.camera1": image_top_tensor,
            "observation.images.camera2": image_wrist_tensor,
            "observation.images.camera3": image_side_tensor,
        }

    observation = {
        **image_observation,
        "observation.state": state_tensor,
        "observation.language.tokens": language_tokens,
        "observation.language.attention_mask": attention_mask,
    }
    
    if debug:
        print(f"\n[Observation Preparation Debug - Manual Fallback]")
        print(f"  Instruction: '{instruction}'")
        print(f"  Policy provided: {policy is not None}")
        if policy is not None:
            print(f"  Has tokenizer: {hasattr(policy, 'tokenizer')}")
            if hasattr(policy, 'tokenizer'):
                print(f"  Token shape: {language_tokens.shape}")
        for key, val in observation.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}")
    
    return observation

# ===== RL Training Helper Functions =====

# Store previous positions for velocity-based rewards
_prev_gripper_pos = None
_prev_block_pos = None
_initial_block_pos = None  # Track where block started (to avoid reward hacking)
_hover_without_contact_steps = 0
_prev_contacted = False
_prev_gripped = False
_prev_block_height = None


def check_gripper_block_contact(m, d, block_name="red_block"):
    """
    Check if gripper or jaw is touching the block.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        block_name: Name of the block body
        
    Returns:
        bool: True if gripper/jaw is in contact with the block
    """
    block_body_id = m.body(block_name).id
    gripper_body_id = m.body("gripper").id
    jaw_body_id = m.body("moving_jaw_so101_v1").id
    
    for i in range(d.ncon):
        contact = d.contact[i]
        geom1_body = m.geom_bodyid[contact.geom1]
        geom2_body = m.geom_bodyid[contact.geom2]
        
        bodies = {geom1_body, geom2_body}
        if block_body_id in bodies:
            if gripper_body_id in bodies or jaw_body_id in bodies:
                return True
    return False


def check_block_gripped_with_force(m, d, block_name="red_block", min_force=0.1):
    """
    Check if block is gripped by analyzing contact forces from both gripper parts.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        block_name: Name of the block body
        min_force: Minimum force threshold (Newtons) to consider as contact
        
    Returns:
        gripped: bool - True if block is being squeezed by both sides
        grip_force: float - magnitude of the weaker grip force (N)
    """
    block_body_id = m.body(block_name).id
    gripper_body_id = m.body("gripper").id
    jaw_body_id = m.body("moving_jaw_so101_v1").id
    
    gripper_force_mag = 0.0
    jaw_force_mag = 0.0
    
    for i in range(d.ncon):
        contact = d.contact[i]
        geom1_body = m.geom_bodyid[contact.geom1]
        geom2_body = m.geom_bodyid[contact.geom2]
        
        bodies = {geom1_body, geom2_body}
        if block_body_id in bodies:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            force_mag = np.linalg.norm(wrench[:3])
            
            if gripper_body_id in bodies:
                gripper_force_mag += force_mag
            if jaw_body_id in bodies:
                jaw_force_mag += force_mag
    
    # Gripped if both sides have sufficient force
    if gripper_force_mag > min_force and jaw_force_mag > min_force:
        return True, min(gripper_force_mag, jaw_force_mag)
    
    return False, 0.0


def get_floor_contact_force(m, d, floor_geom_name="floor"):
    """
    Measure total contact force between robot and floor.
    
    Returns:
        force_magnitude: float - total force magnitude (Newtons)
    """
    floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
    
    total_force = np.zeros(3)
    
    for i in range(d.ncon):
        contact = d.contact[i]
        if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
            # Skip floor-block contacts (we only care about robot-floor)
            block_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "red_block_geom")
            if contact.geom1 == block_geom_id or contact.geom2 == block_geom_id:
                continue
            
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            total_force += wrench[:3]
    
    return np.linalg.norm(total_force)


# Global state for pickup-phase reward tracking
_consecutive_contact = 0


def create_reward_state_tracker() -> dict[str, Any]:
    """Create mutable reward-tracking state for non-global environment wrappers."""
    return {
        "prev_gripper_pos": None,
        "prev_block_pos": None,
        "initial_block_pos": None,
        "consecutive_contact": 0,
        "hover_without_contact_steps": 0,
        "prev_contacted": False,
        "prev_gripped": False,
        "prev_block_height": None,
    }


def _sync_global_reward_state(state: dict[str, Any]) -> None:
    """Mirror a local reward state tracker back into the module globals."""
    global _prev_gripper_pos, _prev_block_pos, _initial_block_pos
    global _consecutive_contact, _hover_without_contact_steps
    global _prev_contacted, _prev_gripped, _prev_block_height

    _prev_gripper_pos = state["prev_gripper_pos"]
    _prev_block_pos = state["prev_block_pos"]
    _initial_block_pos = state["initial_block_pos"]
    _consecutive_contact = state["consecutive_contact"]
    _hover_without_contact_steps = state["hover_without_contact_steps"]
    _prev_contacted = state["prev_contacted"]
    _prev_gripped = state["prev_gripped"]
    _prev_block_height = state["prev_block_height"]


def compute_pickup_reward_from_state(
    m,
    d,
    state: dict[str, Any],
    block_name: str = "red_block",
    lift_threshold: float = 0.08,
    contact_bonus: float = 0.1,
    height_alignment_bonus: float = 0.05,
    grasp_bonus: float = 0.15,
    lift_bonus: float = 0.2,
    lift_bonus_threshold: float = 0.04,
    sustained_contact_threshold: int = 5,
    sustained_contact_bonus: float = 0.2,
):
    """
    Compute a staged pickup reward using caller-owned mutable state.

    Returns:
        reward: scalar reward
        done: success/termination flag for lifting
        metrics: dict with boolean flags and dense shaping diagnostics
    """
    gripper_pos = d.site("gripperframe").xpos.copy()
    block_pos = d.body(block_name).xpos.copy()

    if state["initial_block_pos"] is None:
        state["initial_block_pos"] = block_pos.copy()
    initial_block_pos = state["initial_block_pos"]

    prev_gripper_pos = state["prev_gripper_pos"]
    prev_block_pos = state["prev_block_pos"]
    prev_block_height = state["prev_block_height"]
    prev_contacted = state["prev_contacted"]
    prev_gripped = state["prev_gripped"]

    distance = np.linalg.norm(gripper_pos - block_pos)
    horizontal_dist = np.linalg.norm(gripper_pos[:2] - block_pos[:2])
    height_above = gripper_pos[2] - block_pos[2]
    block_height_gain = max(0.0, block_pos[2] - initial_block_pos[2])
    block_displacement = np.linalg.norm(block_pos[:2] - initial_block_pos[:2])

    prev_horizontal_dist = None
    if prev_gripper_pos is not None and prev_block_pos is not None:
        prev_horizontal_dist = np.linalg.norm(prev_gripper_pos[:2] - prev_block_pos[:2])
    horizontal_progress = 0.0 if prev_horizontal_dist is None else prev_horizontal_dist - horizontal_dist

    reward = -distance

    contacted = check_gripper_block_contact(m, d, block_name)
    gripped, grip_force = check_block_gripped_with_force(m, d, block_name)

    approach_closeness = np.clip(1.0 - horizontal_dist / 0.12, 0.0, 1.0)
    vertical_alignment = np.clip(1.0 - abs(height_above - 0.028) / 0.04, 0.0, 1.0)
    approach_reward = 0.0
    if not contacted:
        approach_reward = 0.04 * approach_closeness + 0.03 * approach_closeness * vertical_alignment
        reward += approach_reward

    moving_into_grasp = horizontal_progress > -0.001
    height_aligned = horizontal_dist < 0.06 and 0.012 < height_above < 0.06 and moving_into_grasp and not contacted
    alignment_reward = 0.0
    if height_aligned:
        alignment_reward = min(0.018, 0.018 * approach_closeness * vertical_alignment)
        reward += alignment_reward

    if height_aligned and not contacted:
        if horizontal_progress < 0.001:
            state["hover_without_contact_steps"] += 1
        else:
            state["hover_without_contact_steps"] = max(0, state["hover_without_contact_steps"] - 1)
    else:
        state["hover_without_contact_steps"] = 0

    hover_stall = state["hover_without_contact_steps"] >= 4
    hover_penalty = -0.02 if hover_stall else 0.0
    reward += hover_penalty

    sustained = False
    contact_entry = False
    contact_entry_bonus = 0.0
    contact_persistence_reward = 0.0
    if contacted:
        state["consecutive_contact"] += 1
        contact_entry = not prev_contacted
        if contact_entry:
            contact_entry_bonus = 0.18
            if height_aligned or horizontal_dist < 0.07:
                contact_entry_bonus += 0.02
        contact_persistence_reward = 0.045
        reward += contact_entry_bonus + contact_persistence_reward
        sustained = state["consecutive_contact"] >= sustained_contact_threshold
        if sustained:
            reward += min(0.06, sustained_contact_bonus * 0.25)
    else:
        state["consecutive_contact"] = 0

    grasp_persistent = False
    grasp_persistence_reward = 0.0
    bilateral_grasp_bonus = 0.0
    if gripped:
        bilateral_grasp_bonus = 0.30
        grasp_persistent = prev_gripped
        grasp_persistence_reward = 0.08 if grasp_persistent else 0.0
        reward += bilateral_grasp_bonus + grasp_persistence_reward

    lift_progress_reward = min(0.4, 5.0 * block_height_gain)
    reward += lift_progress_reward

    block_lifted = block_pos[2] > lift_bonus_threshold
    if block_lifted:
        reward += max(lift_bonus, 0.25)

    slip_count = 0
    slip_penalty = 0.0
    if prev_gripped and not gripped and block_height_gain < 0.02:
        slip_count = 1
        slip_penalty = -0.10
    elif prev_contacted and not contacted and state["consecutive_contact"] == 0:
        slip_count = 1
        slip_penalty = -0.05
    reward += slip_penalty

    block_displacement_penalty = 0.0
    if block_height_gain < 0.015 and block_displacement > 0.02:
        displacement_excess = min(block_displacement - 0.02, 0.08)
        block_displacement_penalty = -0.08 * (displacement_excess / 0.08)
        reward += block_displacement_penalty

    done = block_pos[2] > lift_threshold

    state["prev_gripper_pos"] = gripper_pos
    state["prev_block_pos"] = block_pos
    state["prev_contacted"] = contacted
    state["prev_gripped"] = gripped
    state["prev_block_height"] = block_pos[2]

    metrics = {
        "contacted": contacted,
        "gripped": gripped,
        "sustained": sustained,
        "height_aligned": height_aligned,
        "block_lifted": block_lifted,
        "contact_entry": contact_entry,
        "grasp_persistent": grasp_persistent,
        "hover_stall": hover_stall,
        "slip_count": slip_count,
        "lift_progress": block_height_gain,
        "block_displacement": block_displacement,
        "approach_reward": approach_reward,
        "alignment_reward": alignment_reward,
        "contact_persistence_reward": contact_persistence_reward,
        "grasp_persistence_reward": grasp_persistence_reward,
        "lift_progress_reward": lift_progress_reward,
        "hover_penalty": hover_penalty,
        "slip_penalty": slip_penalty,
        "block_displacement_penalty": block_displacement_penalty,
        "grip_force": grip_force,
        "prev_block_height": prev_block_height if prev_block_height is not None else block_pos[2],
    }
    return reward, done, metrics


def compute_reward(m, d, block_name="red_block", lift_threshold=0.08, contact_bonus=0.1, 
                   height_alignment_bonus=0.05, grasp_bonus=0.15,
                   lift_bonus=0.2, lift_bonus_threshold=0.04,
                   sustained_contact_threshold=5, sustained_contact_bonus=0.2):
    """
    Reward with distance penalty + contact bonus + sustained contact + height alignment + grasp + lift.
    
    Components:
    - Distance: -distance (range: -0.5 to 0.0)
    - Contact bonus: +contact_bonus when gripper touches block
    - Sustained contact: +sustained_contact_bonus after threshold consecutive contact frames
    - Height alignment: +height_alignment_bonus when gripper is above block and close horizontally
    - Grasp bonus: +grasp_bonus when both sides of gripper squeeze block
    - Lift bonus: +lift_bonus when block is elevated above lift_bonus_threshold
    
    Total range per step: ~-0.5 to +0.70

    Returns:
        reward: float - negative distance to block + bonuses
        done: bool - True if block is lifted above lift_threshold (for episode termination)
        contacted: bool - whether gripper is touching block
        gripped: bool - whether both gripper sides are squeezing block
        sustained: bool - whether contact has been sustained above threshold
        height_aligned: bool - whether gripper is above block and close horizontally
        block_lifted: bool - whether block is elevated above lift_bonus_threshold
    """
    state = {
        "prev_gripper_pos": _prev_gripper_pos,
        "prev_block_pos": _prev_block_pos,
        "initial_block_pos": _initial_block_pos,
        "consecutive_contact": _consecutive_contact,
        "hover_without_contact_steps": _hover_without_contact_steps,
        "prev_contacted": _prev_contacted,
        "prev_gripped": _prev_gripped,
        "prev_block_height": _prev_block_height,
    }
    reward, done, metrics = compute_pickup_reward_from_state(
        m,
        d,
        state,
        block_name=block_name,
        lift_threshold=lift_threshold,
        contact_bonus=contact_bonus,
        height_alignment_bonus=height_alignment_bonus,
        grasp_bonus=grasp_bonus,
        lift_bonus=lift_bonus,
        lift_bonus_threshold=lift_bonus_threshold,
        sustained_contact_threshold=sustained_contact_threshold,
        sustained_contact_bonus=sustained_contact_bonus,
    )
    _sync_global_reward_state(state)

    return (
        reward,
        done,
        metrics["contacted"],
        metrics["gripped"],
        metrics["sustained"],
        metrics["height_aligned"],
        metrics["block_lifted"],
        metrics["contact_entry"],
        metrics["grasp_persistent"],
        metrics["lift_progress"],
        metrics["hover_stall"],
        metrics["slip_count"],
        metrics["block_displacement"],
    )


def reset_reward_state():
    """Reset the reward state (call at episode start)."""
    global _prev_gripper_pos, _prev_block_pos, _initial_block_pos, _consecutive_contact
    global _hover_without_contact_steps, _prev_contacted, _prev_gripped, _prev_block_height
    _prev_gripper_pos = None
    _prev_block_pos = None
    _initial_block_pos = None
    _consecutive_contact = 0
    _hover_without_contact_steps = 0
    _prev_contacted = False
    _prev_gripped = False
    _prev_block_height = None


def reset_env(m, d, starting_position, block_pos=(0, 0.3, 0.0125)):
    """
    Reset robot and block to initial positions.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        starting_position: dict with joint positions in degrees
        block_pos: tuple (x, y, z) for block initial position
    """
    # Reset all state
    mujoco.mj_resetData(m, d)
    
    # Set robot to starting pose
    set_initial_pose(d, starting_position)
    
    # Reset block position (qpos indices after robot joints)
    # Block has freejoint: 3 pos + 4 quat = 7 DOF
    # Robot has 6 joints, so block starts at index 6
    d.qpos[6:9] = block_pos  # position (x, y, z)
    d.qpos[9:13] = [1, 0, 0, 0]  # quaternion (w, x, y, z) - upright
    
    # Zero out velocities
    d.qvel[:] = 0
    
    # Forward kinematics to update positions
    mujoco.mj_forward(m, d)
