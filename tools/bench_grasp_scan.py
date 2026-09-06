"""Simulation-only depth-lead diagnostic for the bench teacher (moved from notes/handoff-support, 2026-09-06).

Runs the active bench scene at the nominal pose with several depth leads and prints strict-grasp
frames, peak cube height, and the terminal result. Teacher tuning only; it promotes nothing.
Run from the Git root with PYTHONNOUSERSITE=1 MUJOCO_GL=egl.
"""
from pathlib import Path
import json
from dataclasses import asdict
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.bench import bench_suite
from so_arm101_v2.simulation.privileged import PrivilegedStagedController
from so_arm101_v2.contracts.pick_place import load_pick_place_contract,PickPlaceEvaluationState,evaluate_pick_place_step
p=Path('simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
for lead in [0.,-.003,-.006,-.009,.003]:
 a=MujocoTaskAdapter(p);a.reset(bench_suite(a.bench,[(0,0)],label='diag',repeats=1).scenarios[0]);c=PrivilegedStagedController(depth_lead_m=lead);c.reset(a);state=PickPlaceEvaluationState();contract=load_pick_place_contract('bench_pick_replace_v1',bench_config=a.bench);strict=0;maxhold=0;maxheight=0
 for i in range(480):
  cmd=a.apply_policy_command(c.predict(None,a.current_act(),a));a.advance_control_period();m,_=a.pick_place_measurement(cmd);state,result=evaluate_pick_place_step(contract,m,state);strict+=m.pickup.strict_bilateral_grasp;maxheight=max(maxheight,m.pickup.cube_height_gain_m)
  if result.terminated or result.truncated or result.invalidated:break
 print(json.dumps(dict(lead=lead,strict=strict,maxheight=maxheight,step=i,result=asdict(result))),flush=True);a.close()
