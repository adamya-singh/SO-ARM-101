"""Teacher jaw-cube distance per row (mm), clean SO-101 capture vs perturbed SO-101 capture, first 600 episodes."""
import glob, numpy as np
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
ROWS = [90, 120, 150, 179, 190, 200, 210, 220, 240, 250, 270]
print("capture".ljust(10) + "".join(str(r).rjust(7) for r in ROWS))
for name, c in [("clean", "so101_benchmark_20261106/so101/cap4800"), ("perturbed", "perturbed_so101_20261108/capture7200")]:
    z = np.load(glob.glob(f"{E}/{c}/capture_train/oracle/*/*/demonstrations.npz")[0])
    d = z["post_jaw_cube_distance_m"][: 600 * 480].reshape(600, 480) * 1000
    print(name.ljust(10) + "".join(f"{np.median(d[:, r]):7.1f}" for r in ROWS))
    u = z["post_unsafe_contact"][: 600 * 480].reshape(600, 480)
    print("  episodes with unsafe contact:", int(u.any(1).sum()), "/600")
