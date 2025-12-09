"""
LeRobot Dataset Writer for v3 Format

Writes teleoperation demonstrations in LeRobotDataset v3.0 format,
compatible with SmolVLA fine-tuning.

Directory structure:
    dataset_name/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.parquet
    │   └── stats.json
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet
    └── videos/
        ├── observation.images.camera1/
        │   └── chunk-000/
        │       └── episode_000000.mp4
        ├── observation.images.camera2/
        │   └── chunk-000/
        │       └── episode_000000.mp4
        └── observation.images.camera3/
            └── chunk-000/
                └── episode_000000.mp4
"""

import os
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import imageio


class LeRobotDatasetWriter:
    """
    Writer for LeRobotDataset v3.0 format.
    
    Records observations, actions, and metadata from teleoperation
    sessions and saves in the format expected by LeRobot/SmolVLA.
    """
    
    def __init__(
        self,
        output_dir: str,
        task_description: str,
        fps: int = 10,
        image_size: tuple = (256, 256),
        repo_id: Optional[str] = None,
        robot_type: str = "so101",
    ):
        """
        Initialize the dataset writer.
        
        Args:
            output_dir: Directory to save the dataset
            task_description: Description of the task being recorded
            fps: Frames per second for recording
            image_size: (height, width) of camera images
            repo_id: Hugging Face repo ID (e.g., "username/dataset_name")
            robot_type: Type of robot being used
        """
        self.output_dir = Path(output_dir)
        self.task_description = task_description
        self.fps = fps
        self.image_size = image_size
        self.repo_id = repo_id or self.output_dir.name
        self.robot_type = robot_type
        
        # Create directory structure
        self._setup_directories()
        
        # Episode tracking - check for existing episodes to append
        self.current_episode_idx = self._find_next_episode_idx()
        self.current_frames: List[Dict[str, Any]] = []
        self.all_episodes_metadata: List[Dict[str, Any]] = []
        
        # Load existing episode metadata if appending
        self._existing_episode_count = self.current_episode_idx
        if self.current_episode_idx > 0:
            print(f"Found {self.current_episode_idx} existing episodes, will append new episodes...")
            self._load_existing_metadata()
        
        # Video writers for current episode
        self._video_writers: Dict[str, Any] = {}
        
        # Statistics accumulators
        self._stats_accum: Dict[str, Dict[str, List]] = {
            'observation.state': {'values': []},
            'action': {'values': []},
        }
        
        # Camera keys
        self.camera_keys = [
            'observation.images.camera1',
            'observation.images.camera2',
            'observation.images.camera3',
        ]
    
    def _setup_directories(self):
        """Create the dataset directory structure."""
        # Main directories
        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        
        # Video directories for each camera
        for camera_key in ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']:
            (self.output_dir / "videos" / camera_key / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    def _find_next_episode_idx(self) -> int:
        """Find the next available episode index by scanning existing files."""
        data_dir = self.output_dir / "data" / "chunk-000"
        if not data_dir.exists():
            return 0
        
        existing = list(data_dir.glob("episode_*.parquet"))
        if not existing:
            return 0
        
        # Extract indices and find max
        indices = []
        for f in existing:
            # episode_000042.parquet -> 42
            try:
                idx = int(f.stem.split("_")[1])
                indices.append(idx)
            except (IndexError, ValueError):
                continue
        
        if not indices:
            return 0
        
        return max(indices) + 1
    
    def _load_existing_metadata(self):
        """Load existing episode metadata when appending to a dataset."""
        episodes_file = self.output_dir / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
        if not episodes_file.exists():
            return
        
        try:
            table = pq.read_table(episodes_file)
            df = table.to_pandas()
            
            for _, row in df.iterrows():
                self.all_episodes_metadata.append({
                    'episode_index': int(row['episode_index']),
                    'num_frames': int(row['num_frames']),
                    'length_s': float(row['length_s']),
                    'task_index': int(row['task_index']),
                    'success': True,  # Assume existing episodes were successful
                })
            
            # Also load existing stats for merging
            stats_file = self.output_dir / "meta" / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    existing_stats = json.load(f)
                # We'll recompute stats from all data during finalize
                
        except Exception as e:
            print(f"Warning: Could not load existing metadata: {e}")
    
    def start_episode(self):
        """Start recording a new episode."""
        self.current_frames = []
        self._video_writers = {}
        
        # Initialize video writers for each camera
        for camera_key in self.camera_keys:
            video_path = self._get_video_path(camera_key, self.current_episode_idx)
            self._video_writers[camera_key] = imageio.get_writer(
                video_path,
                fps=self.fps,
                codec='libx264',
                quality=8,  # 0-10, higher is better
                pixelformat='yuv420p',
            )
        
        print(f"Started recording episode {self.current_episode_idx}")
    
    def add_frame(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        timestamp: float,
    ):
        """
        Add a frame to the current episode.
        
        Args:
            observation: Dict with camera images and state
            action: Action taken (6,) joint positions
            timestamp: Time since episode start
        """
        frame_idx = len(self.current_frames)
        
        # Write camera images to video
        for camera_key in self.camera_keys:
            if camera_key in observation:
                image = observation[camera_key]
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                self._video_writers[camera_key].append_data(image)
        
        # Store frame data (without images, those go to video)
        frame_data = {
            'frame_index': frame_idx,
            'episode_index': self.current_episode_idx,
            'timestamp': timestamp,
            'observation.state': observation['observation.state'].astype(np.float32),
            'action': action.astype(np.float32),
            'task_index': 0,  # Single task
        }
        self.current_frames.append(frame_data)
        
        # Accumulate stats
        self._stats_accum['observation.state']['values'].append(
            observation['observation.state'].astype(np.float32)
        )
        self._stats_accum['action']['values'].append(action.astype(np.float32))
    
    def end_episode(self, success: bool = True) -> int:
        """
        End the current episode and save data.
        
        Args:
            success: Whether the episode was successful
            
        Returns:
            Number of frames recorded in this episode
        """
        if not self.current_frames:
            print("No frames recorded, skipping episode save")
            return 0
        
        num_frames = len(self.current_frames)
        
        # Close video writers
        for camera_key, writer in self._video_writers.items():
            writer.close()
        self._video_writers = {}
        
        # Save episode data to parquet
        self._save_episode_parquet()
        
        # Record episode metadata
        episode_meta = {
            'episode_index': self.current_episode_idx,
            'num_frames': num_frames,
            'length_s': num_frames / self.fps,
            'success': success,
            'task_index': 0,
        }
        self.all_episodes_metadata.append(episode_meta)
        
        print(f"Saved episode {self.current_episode_idx} with {num_frames} frames")
        
        # Increment episode counter
        self.current_episode_idx += 1
        self.current_frames = []
        
        return num_frames
    
    def _get_video_path(self, camera_key: str, episode_idx: int) -> str:
        """Get the video file path for a camera and episode."""
        return str(
            self.output_dir / "videos" / camera_key / "chunk-000" / 
            f"episode_{episode_idx:06d}.mp4"
        )
    
    def _save_episode_parquet(self):
        """Save current episode frames to parquet file."""
        if not self.current_frames:
            return
        
        # Get dimensions for fixed-length arrays
        state_dim = self.current_frames[0]['observation.state'].shape[0]
        action_dim = self.current_frames[0]['action'].shape[0]
        
        # Define explicit schema with fixed-length float32 arrays
        # This is critical for LeRobot compatibility
        schema = pa.schema([
            ('frame_index', pa.int64()),
            ('episode_index', pa.int64()),
            ('timestamp', pa.float64()),
            ('task_index', pa.int64()),
            ('observation.state', pa.list_(pa.float32(), state_dim)),
            ('action', pa.list_(pa.float32(), action_dim)),
        ])
        
        # Create arrays with explicit types
        arrays = [
            pa.array([f['frame_index'] for f in self.current_frames], type=pa.int64()),
            pa.array([f['episode_index'] for f in self.current_frames], type=pa.int64()),
            pa.array([f['timestamp'] for f in self.current_frames], type=pa.float64()),
            pa.array([f['task_index'] for f in self.current_frames], type=pa.int64()),
            pa.array([f['observation.state'].tolist() for f in self.current_frames], 
                     type=pa.list_(pa.float32(), state_dim)),
            pa.array([f['action'].tolist() for f in self.current_frames], 
                     type=pa.list_(pa.float32(), action_dim)),
        ]
        
        # Create table with explicit schema
        table = pa.table(dict(zip(schema.names, arrays)), schema=schema)
        
        # Save to parquet
        parquet_path = (
            self.output_dir / "data" / "chunk-000" / 
            f"episode_{self.current_episode_idx:06d}.parquet"
        )
        pq.write_table(table, parquet_path)
    
    def finalize(self):
        """Finalize the dataset and write metadata files."""
        if not self.all_episodes_metadata:
            print("No episodes recorded!")
            return
        
        # Calculate statistics (only from new episodes in this session)
        stats = self._compute_stats()
        
        # Write info.json
        self._write_info_json()
        
        # Write tasks.parquet
        self._write_tasks_parquet()
        
        # Write stats.json (note: stats are only from this session's episodes)
        self._write_stats_json(stats)
        
        # Write episodes metadata (includes all episodes: existing + new)
        self._write_episodes_parquet()
        
        total_frames = sum(ep['num_frames'] for ep in self.all_episodes_metadata)
        new_episodes = len(self.all_episodes_metadata) - self._existing_episode_count
        
        print(f"\nDataset finalized:")
        print(f"  Total episodes: {len(self.all_episodes_metadata)}")
        if self._existing_episode_count > 0:
            print(f"    - Existing: {self._existing_episode_count}")
            print(f"    - New: {new_episodes}")
        print(f"  Total frames: {total_frames}")
        print(f"  Output directory: {self.output_dir}")
    
    def _compute_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute dataset statistics for normalization."""
        stats = {}
        
        for key, accum in self._stats_accum.items():
            if not accum['values']:
                continue
            
            values = np.array(accum['values'])
            stats[key] = {
                'mean': values.mean(axis=0).tolist(),
                'std': values.std(axis=0).tolist(),
                'min': values.min(axis=0).tolist(),
                'max': values.max(axis=0).tolist(),
            }
        
        return stats
    
    def _write_info_json(self):
        """Write dataset info.json metadata file."""
        total_frames = sum(ep['num_frames'] for ep in self.all_episodes_metadata)
        total_episodes = len(self.all_episodes_metadata)
        
        info = {
            'codebase_version': '3.0',
            'robot_type': self.robot_type,
            'fps': self.fps,
            'total_episodes': total_episodes,
            'total_frames': total_frames,
            'total_tasks': 1,
            'total_videos': total_episodes * len(self.camera_keys),
            'total_chunks': 1,
            'chunks_size': total_episodes,
            'data_path': 'data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet',
            # Use file_index instead of episode_index for video path (LeRobot v3 requirement)
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4',
            'features': {
                # Metadata columns (required by LeRobot v3)
                'timestamp': {
                    'dtype': 'float64',
                    'shape': [1],
                },
                'frame_index': {
                    'dtype': 'int64',
                    'shape': [1],
                },
                'episode_index': {
                    'dtype': 'int64',
                    'shape': [1],
                },
                'task_index': {
                    'dtype': 'int64',
                    'shape': [1],
                },
                # Video features
                'observation.images.camera1': {
                    'dtype': 'video',
                    'shape': [self.image_size[0], self.image_size[1], 3],
                    'names': ['height', 'width', 'channels'],
                    'video_info': {
                        'video.fps': self.fps,
                        'video.codec': 'libx264',
                        'video.pix_fmt': 'yuv420p',
                    }
                },
                'observation.images.camera2': {
                    'dtype': 'video',
                    'shape': [self.image_size[0], self.image_size[1], 3],
                    'names': ['height', 'width', 'channels'],
                    'video_info': {
                        'video.fps': self.fps,
                        'video.codec': 'libx264',
                        'video.pix_fmt': 'yuv420p',
                    }
                },
                'observation.images.camera3': {
                    'dtype': 'video',
                    'shape': [self.image_size[0], self.image_size[1], 3],
                    'names': ['height', 'width', 'channels'],
                    'video_info': {
                        'video.fps': self.fps,
                        'video.codec': 'libx264',
                        'video.pix_fmt': 'yuv420p',
                    }
                },
                # State and action features
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [6],
                    'names': ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 
                              'wrist_flex', 'wrist_roll', 'gripper'],
                },
                'action': {
                    'dtype': 'float32',
                    'shape': [6],
                    'names': ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                              'wrist_flex', 'wrist_roll', 'gripper'],
                },
            },
            'repo_id': self.repo_id,
            'created_at': datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "meta" / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def _write_tasks_parquet(self):
        """Write tasks.parquet file."""
        tasks_data = {
            'task_index': [0],
            'task': [self.task_description],
        }
        table = pa.table(tasks_data)
        pq.write_table(table, self.output_dir / "meta" / "tasks.parquet")
    
    def _write_stats_json(self, stats: Dict[str, Dict[str, Any]]):
        """Write stats.json file."""
        # Add ImageNet normalization stats for camera features (required by LeRobot v3)
        # These are placeholder stats since videos use ImageNet normalization
        for camera_key in self.camera_keys:
            stats[camera_key] = {
                'mean': [[[0.485]], [[0.456]], [[0.406]]],
                'std': [[[0.229]], [[0.224]], [[0.225]]],
                'min': [[[0.0]], [[0.0]], [[0.0]]],
                'max': [[[1.0]], [[1.0]], [[1.0]]]
            }
        
        with open(self.output_dir / "meta" / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _write_episodes_parquet(self):
        """Write episodes metadata to parquet."""
        episodes_dir = self.output_dir / "meta" / "episodes" / "chunk-000"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        
        num_episodes = len(self.all_episodes_metadata)
        
        # Calculate dataset_from_index and dataset_to_index
        # These represent the cumulative frame indices for each episode
        dataset_from = []
        dataset_to = []
        current_idx = 0
        for ep in self.all_episodes_metadata:
            dataset_from.append(current_idx)
            current_idx += ep['num_frames']
            dataset_to.append(current_idx)
        
        episodes_data = {
            # Basic episode metadata
            'episode_index': [ep['episode_index'] for ep in self.all_episodes_metadata],
            'num_frames': [ep['num_frames'] for ep in self.all_episodes_metadata],
            'length_s': [ep['length_s'] for ep in self.all_episodes_metadata],
            'task_index': [ep['task_index'] for ep in self.all_episodes_metadata],
            
            # Dataset frame indices (required by LeRobot v3)
            'dataset_from_index': dataset_from,
            'dataset_to_index': dataset_to,
        }
        
        # Add video metadata columns for each camera (required by LeRobot v3)
        for camera_key in self.camera_keys:
            # Video timestamps: each episode's video starts at 0 and ends at length_s
            episodes_data[f'videos/{camera_key}/from_timestamp'] = [0.0] * num_episodes
            episodes_data[f'videos/{camera_key}/to_timestamp'] = [ep['length_s'] for ep in self.all_episodes_metadata]
            # All videos are in chunk 0
            episodes_data[f'videos/{camera_key}/chunk_index'] = [0] * num_episodes
            # File index equals episode index (one video file per episode)
            episodes_data[f'videos/{camera_key}/file_index'] = [ep['episode_index'] for ep in self.all_episodes_metadata]
        
        table = pa.table(episodes_data)
        pq.write_table(table, episodes_dir / "episodes.parquet")


def test_dataset_writer():
    """Test the dataset writer with dummy data."""
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        writer = LeRobotDatasetWriter(
            output_dir=os.path.join(temp_dir, "test_dataset"),
            task_description="Test task: pick up the block",
            fps=10,
        )
        
        # Record a dummy episode
        writer.start_episode()
        
        for i in range(50):  # 5 seconds at 10 fps
            obs = {
                'observation.images.camera1': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                'observation.images.camera2': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                'observation.images.camera3': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                'observation.state': np.random.randn(6).astype(np.float32),
            }
            action = np.random.randn(6).astype(np.float32)
            timestamp = i / 10.0
            
            writer.add_frame(obs, action, timestamp)
        
        writer.end_episode(success=True)
        writer.finalize()
        
        print("\nTest passed! Dataset structure:")
        for path in sorted(Path(temp_dir).rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(temp_dir)}")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_dataset_writer()

