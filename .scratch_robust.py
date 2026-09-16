import time
import numpy as np
from voxhands.mujoco_workcell import DinnerTableWorkcell
from voxhands.assets.scene_dinner import DinnerConfig, OBJECT_REGISTRY


def seed_config(seed: int) -> DinnerConfig:
    rng = np.random.default_rng(seed)
    cfg = DinnerConfig(seed=seed)
    for oid in OBJECT_REGISTRY:
        cfg.object_yaws[oid] = float(rng.uniform(-np.pi, np.pi))
    homes = {"plate": (0.26, 0.60), "cup": (0.74, 0.60), "spoon": (0.40, 0.76),
             "fork": (0.60, 0.76), "bottle": (0.30, 0.86)}
    for oid, (hx, hy) in homes.items():
        cfg.object_offset[oid] = (float(rng.uniform(-0.012, 0.012)), float(rng.uniform(-0.012, 0.012)))
    for oid in OBJECT_REGISTRY:
        cfg.object_scales[oid] = float(rng.uniform(0.96, 1.04))
        cfg.object_masses[oid] = float(rng.uniform(0.8, 1.2)) * OBJECT_REGISTRY[oid]["mass"]
    return cfg


def scenario(env) -> list[tuple[str, bool, str]]:
    r = []
    def rep(name, res):
        r.append((name, res.success, res.message))
    rep("open_drawer", env.open_drawer("left"))
    rep("grasp_plate", env.grasp("left", "plate"))
    rep("place_plate", env.place("left", "plate", "left_place"))
    rep("grasp_bottle", env.grasp("left", "bottle"))
    rep("grasp_cup", env.grasp("right", "cup"))
    rep("pour", env.pour("left", "right", "bottle", "cup"))
    rep("place_cup", env.place("right", "cup", "right_place"))
    rep("place_bottle", env.place("left", "bottle", "center_place"))
    return r


tot = time.perf_counter()
fails = {}
for seed in range(10):
    env = DinnerTableWorkcell(seed_config(seed), oracle=True)
    t0 = time.perf_counter()
    r = scenario(env)
    ok = sum(1 for _, s, _ in r if s)
    bad = [n for n, s, _ in r if not s]
    fails[seed] = bad
    print(f"seed {seed:2d}  {ok}/{len(r)} ok  {time.perf_counter()-t0:5.2f}s  bad={bad}")
print(f"TOTAL {time.perf_counter()-tot:.1f}s")
