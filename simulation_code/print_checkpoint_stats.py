import torch
import numpy as np

checkpoint = torch.load("reinflow_checkpoint.pt", weights_only=False)

rewards = checkpoint['episode_rewards']
sigmas = checkpoint['log_sigmas'].exp().numpy()

print(f"Loaded checkpoint from episode {checkpoint['episode']}")
print(f"σ_mean at end: {sigmas.mean():.4f}")
print(f"\nReplay of training stats:\n")

for i in range(9, len(rewards), 10):  # Every 10th episode (0-indexed: 9, 19, 29...)
    episode = i + 1
    reward = rewards[i]
    avg_reward = np.mean(rewards[max(0, i-9):i+1])  # Last 10 episodes
    
    print(f"Episode {episode:5d} | Reward: {reward:8.2f} | Avg: {avg_reward:8.2f}")
    
    if episode % 100 == 0:
        print(f"[ReinFlow] Checkpoint saved to reinflow_checkpoint.pt")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Total episodes: {len(rewards)}")
print(f"  Best reward: {max(rewards):.2f}")
print(f"  First 100 avg: {np.mean(rewards[:100]):.2f}")
print(f"  Last 100 avg: {np.mean(rewards[-100:]):.2f}")
