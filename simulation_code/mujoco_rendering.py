"""
MuJoCo rendering backend configuration.
Import this BEFORE importing mujoco to enable headless rendering.

Usage:
    from mujoco_rendering import setup_mujoco_rendering
    setup_mujoco_rendering()  # Call BEFORE importing mujoco
    
    import mujoco
    import mujoco.viewer
"""
import os
import sys


def setup_mujoco_rendering(prefer_headless=False, verbose=True):
    """
    Configure MuJoCo rendering backend with automatic fallback.
    
    Priority (if prefer_headless=True or no display):
      1. EGL (GPU headless)
      2. OSMesa (software headless)
      3. Native OpenGL (requires display)
    
    Must be called BEFORE importing mujoco.
    
    Args:
        prefer_headless: If True, use headless backend even if display is available
        verbose: If True, print which backend is being used
    
    Returns:
        str or None: The backend being used ('egl', 'osmesa', or None for native)
    """
    # Skip if already configured
    if 'MUJOCO_GL' in os.environ:
        if verbose:
            print(f"Using MUJOCO_GL={os.environ['MUJOCO_GL']} (pre-configured)")
        return os.environ['MUJOCO_GL']
    
    # Check if display is available
    display_available = os.environ.get('DISPLAY') is not None
    
    # On macOS, we don't need EGL/OSMesa - native rendering works
    if sys.platform == 'darwin':
        if verbose:
            print("Using native OpenGL rendering (macOS)")
        return None
    
    if display_available and not prefer_headless:
        if verbose:
            print("Using native OpenGL rendering (display available)")
        return None  # Use default
    
    # Try EGL first (GPU headless)
    if _test_egl_available():
        os.environ['MUJOCO_GL'] = 'egl'
        if verbose:
            print("Using EGL rendering (headless GPU)")
        return 'egl'
    
    # Fall back to OSMesa (software)
    if _test_osmesa_available():
        os.environ['MUJOCO_GL'] = 'osmesa'
        if verbose:
            print("Using OSMesa rendering (headless software)")
        return 'osmesa'
    
    # No headless backend available
    if verbose:
        print("WARNING: No headless rendering backend available. "
              "Install libosmesa6-dev for software rendering or ensure EGL is available.")
    return None


def _test_egl_available():
    """Check if EGL is available."""
    try:
        import ctypes
        ctypes.CDLL('libEGL.so.1')
        return True
    except OSError:
        return False


def _test_osmesa_available():
    """Check if OSMesa is available."""
    try:
        import ctypes
        ctypes.CDLL('libOSMesa.so')
        return True
    except OSError:
        return False


# Auto-setup when module is imported (can be overridden by calling setup_mujoco_rendering manually first)
# This allows simple usage: `from mujoco_rendering import setup_mujoco_rendering` followed by mujoco import

