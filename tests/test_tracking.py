"""The W&B standard (notes/wandb-standard.md) as enforced by so_arm101_v2.tracking."""

import json

import pytest

from so_arm101_v2 import tracking


def recording(**kwargs):
    return tracking.init_run(family="rl-act", name=kwargs.pop("name", "rl-act-rehearsal"), config={}, job_type="rl",
                             rehearsal=True, enabled=False, **kwargs)


def test_numbered_keys_pass_and_unnumbered_keys_raise_in_strict_runs():
    tracker = recording()
    tracker.log({"01_outcome/success_rate": 0.5, "03_training/step": 3})
    with pytest.raises(ValueError):
        tracker.log({"reward/ema20": 1.0})
    with pytest.raises(ValueError):
        tracker.log({"06_extra/x": 1.0})


def test_real_runs_drop_unnumbered_keys_instead_of_stopping():
    tracker = tracking.Tracker(None, strict=False)
    tracker.log({"heldout_fit/ratio": 2.0, "02_generalisation/ratio_start_90": 2.0})
    assert tracker.logged == {"02_generalisation/ratio_start_90"}


def test_summary_must_name_a_logged_key():
    tracker = recording()
    tracker.define("02_generalisation/per_pose_median", summary="min")
    tracker.log({"02_generalisation/median": 1e-4})
    assert tracker.unlogged_summaries() == ["02_generalisation/per_pose_median"]
    with pytest.raises(ValueError):
        tracker.finish()
    with pytest.raises(ValueError):
        tracker.define("02_generalisation/*", summary="min")


def test_rehearsal_runs_must_be_named_rehearsal_and_job_types_are_fixed():
    with pytest.raises(ValueError):
        recording(name="rl-act-K4")
    with pytest.raises(ValueError):
        tracking.init_run(family="x", name="x-rehearsal", config={}, job_type="train", rehearsal=True, enabled=False)


def test_run_name_and_reference_lines():
    assert tracking.run_name("rl-pi05", "K10", "n8000", scene_hash="92f07142abcdef", seed=202) == "rl-pi05-K10-n8000-92f07142-seed202"
    assert tracking.run_name("rl-pi05", "K10", rehearsal=True) == "rl-pi05-rehearsal"
    for key in tracking.REFERENCE_LINES:
        tracking.check_key(key)


def test_phase_and_reference_logging_use_numbered_keys(tmp_path):
    tracker = recording()
    tracker.phase("train", ("load", "train", "complete"), 5, axis=tracking.EPISODES)
    tracker.references(5)
    assert tracker.records[0]["05_phases/phase"] == 1 and tracker.records[0]["05_phases/train"] == 1
    assert all(tracking.KEY.match(k) for record in tracker.records for k in record)
    tracking.persist_run_id(tmp_path / "wandb.json", tracker)
    assert json.loads((tmp_path / "wandb.json").read_text())["project"] == tracking.PROJECT
