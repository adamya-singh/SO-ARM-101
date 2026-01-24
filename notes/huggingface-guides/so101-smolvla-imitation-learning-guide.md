# Imitation Learning on Real-World Robots

This tutorial will explain how to train a neural network to control a real robot autonomously.

link: https://huggingface.co/docs/lerobot/en/il_robots

**You'll learn:**

1. How to record and visualize your dataset.
2. How to train a policy using your data and prepare it for evaluation.
3. How to evaluate your policy and visualize the results.

By following these steps, you'll be able to replicate tasks, such as picking up a Lego block and placing it in a bin with a high success rate, as shown in the video below.

Video: pickup lego block task

  
    
  

This tutorial isn’t tied to a specific robot: we walk you through the commands and API snippets you can adapt for any supported platform.

During data collection, you’ll use a “teloperation” device, such as a leader arm or keyboard to teleoperate the robot and record its motion trajectories.

Once you’ve gathered enough trajectories, you’ll train a neural network to imitate these trajectories and deploy the trained model so your robot can perform the task autonomously.

If you run into any issues at any point, jump into our [Discord community](https://discord.com/invite/s3KuuzsPFb) for support.

## Set up and Calibrate

If you haven't yet set up and calibrated your robot and teleop device, please do so by following the robot-specific tutorial.

## Teleoperate

In this example, we’ll demonstrate how to teleoperate the SO101 robot. For each command, we also provide a corresponding API example.

Note that the `id` associated with a robot is used to store the calibration file. It's important to use the same `id` when teleoperating, recording, and evaluating when using the same setup.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm
```

```python
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem58760431541",
    id="my_red_robot_arm",
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_blue_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)
```

The teleoperate command will automatically:

1. Identify any missing calibrations and initiate the calibration procedure.
2. Connect the robot and teleop device and start teleoperation.

## Cameras

To add cameras to your setup, follow this [Guide](./cameras#setup-cameras).

## Teleoperate with cameras

With `rerun`, you can teleoperate again while simultaneously visualizing the camera feeds and joint positions. In this example, we’re using the Koch arm.

```bash
lerobot-teleoperate \
    --robot.type=koch_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=koch_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
}

robot_config = KochFollowerConfig(
    port="/dev/tty.usbmodem585A0076841",
    id="my_red_robot_arm",
    cameras=camera_config
)

teleop_config = KochLeaderConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_blue_leader_arm",
)

robot = KochFollower(robot_config)
teleop_device = KochLeader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)
```

## Record a dataset

Once you're familiar with teleoperation, you can record your first dataset.

We use the Hugging Face hub features for uploading your dataset. If you haven't previously used the Hub, make sure you can login via the cli using a write-access token, this token can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens).

Add your token to the CLI by running this command:

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then store your Hugging Face repository name in a variable:

```bash
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```

Now you can record a dataset. To record 5 episodes and upload your dataset to the hub, adapt the code below for your robot and execute the command or API example.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube"
```

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

# Create robot configuration
robot_config = SO100FollowerConfig(
    id="my_awesome_follower_arm",
    cameras={
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS) # Optional: fourcc="MJPG" for troubleshooting OpenCV async error.
    },
    port="/dev/tty.usbmodem58760434471",
)

teleop_config = SO100LeaderConfig(
    id="my_awesome_leader_arm",
    port="/dev/tty.usbmodem585A0077581",
)

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)
teleop = SO100Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="/",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
    
    # Run the recording loop
    record_loop(
        robot=robot,
        teleop=teleop,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        reset_time_s=RESET_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )
    
    dataset.save_episode()
    episode_idx += 1

# Clean up
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
```

Your robot should replicate movements similar to those you recorded. For example, check out [this video](https://x.com/RemiCadene/status/1793654950905680090) where we use `replay` on a Aloha robot from [Trossen Robotics](https://www.trossenrobotics.com).

## Train a policy

To train a policy to control your robot, use the [`lerobot-train`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py) script. A few arguments are required. Here is an example command:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

Let's explain the command:

1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/so101_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor states, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
3. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
4. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_so101_test/checkpoints`.

To resume training from a checkpoint, below is an example command to resume from `last` checkpoint of the `act_so101_test` policy:

```bash
lerobot-train \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

If you do not want to push your model to the hub after training use `--policy.push_to_hub=false`.

Additionally you can provide extra `tags` or specify a `license` for your model or make the model repo `private` by adding this: `--policy.private=true --policy.tags=\[ppo,rl\] --policy.license=mit`

#### Train using Google Colab

If your local computer doesn't have a powerful GPU you could utilize Google Colab to train your model by following the [ACT training notebook](./notebooks#training-act).

#### Upload policy checkpoints

Once training is done, upload the latest checkpoint with:

```bash
huggingface-cli upload ${HF_USER}/act_so101_test \
  outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

You can also upload intermediate checkpoints with:

```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/act_so101_test${CKPT} \
  outputs/train/act_so101_test/checkpoints/${CKPT}/pretrained_model
```

## Run inference and evaluate your policy

You can use the `record` script from [`lerobot-record`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_record.py) with a policy checkpoint as input, to run inference and evaluate your policy. For instance, run this command or API example to run inference and record 10 evaluation episodes:

```bash
lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_so100 \
  --dataset.single_task="Put lego brick into the transparent box" \
  --control.policy.path=${HF_USER}/act_so101_test \
  --dataset.num_episodes=10
```

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.policies.act.policy_act import ACTPolicy
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_pre_post_processors

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Put lego brick into the transparent box"
HF_MODEL_ID = "${HF_USER}/act_so101_test"
HF_DATASET_ID = "${HF_USER}/eval_so100"

# Create the robot configuration
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm", cameras=camera_config
)

# Initialize the robot
robot = SO100Follower(robot_config)

# Initialize the policy
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot
robot.connect()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    dataset.save_episode()

# Clean up
robot.disconnect()
dataset.push_to_hub()
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:

1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with (e.g. `outputs/train/eval_act_so101_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_so101_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_so101_test`).

