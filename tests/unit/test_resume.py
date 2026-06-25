"""True crash-resume: trainer must restore optimizer/scheduler/epoch/step, not just weights.

Bug: Trainer._load_checkpoint did `model.load_state_dict(strict=False)` only — a weights-only
warm-start (correct for cross-phase P1→P2). It ignored optimizer/scheduler/epoch/step, so a
preempted Modal container relaunched from epoch 0 with a fresh optimizer + restarted LR schedule.
A 20-epoch run can't be safely chopped across containers without this.

These pin the two pure helpers the resume wiring uses + the save/load round-trip it relies on.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.trainer import Trainer
from src.utils.common import load_checkpoint, save_checkpoint


# ---- _resume_state: (start_epoch, step, best_metric) from a loaded checkpoint dict ----

def test_resume_state_advances_past_saved_epoch():
    # saved epoch already completed → resume at epoch+1; step/best_metric carry over verbatim.
    assert Trainer._resume_state({"epoch": 7, "step": 1234, "best_metric": 0.42}) == (8, 1234, 0.42)


# ---- max_epochs_per_run: cap each detached launch without breaking resume ----

def test_normalize_max_epochs_per_run_accepts_unset_or_positive_values():
    assert Trainer._normalize_max_epochs_per_run(None) is None
    assert Trainer._normalize_max_epochs_per_run("3") == 3
    assert Trainer._normalize_max_epochs_per_run(1) == 1


def test_normalize_max_epochs_per_run_rejects_non_positive_values():
    for value in (0, -1):
        try:
            Trainer._normalize_max_epochs_per_run(value)
        except ValueError as exc:
            assert "max_epochs_per_run must be > 0" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {value}")


def test_reached_max_epochs_per_run_counts_from_resume_start_epoch():
    # First launch: start at 0, cap=2 → stop after epochs 0 and 1.
    assert not Trainer._reached_max_epochs_per_run(start_epoch=0, epoch=0, max_per_run=2)
    assert Trainer._reached_max_epochs_per_run(start_epoch=0, epoch=1, max_per_run=2)

    # Relaunch after resume: saved epoch_1 means start_epoch=2; cap=2 stops after 2 and 3.
    assert not Trainer._reached_max_epochs_per_run(start_epoch=2, epoch=2, max_per_run=2)
    assert Trainer._reached_max_epochs_per_run(start_epoch=2, epoch=3, max_per_run=2)


# ---- auxiliary loss ramp: force-visual training should switch on predictably ----

def test_ramped_aux_weight_is_zero_before_start_epoch():
    assert Trainer._ramped_aux_weight(epoch=2, base_weight=0.5, warmup_epochs=2, start_epoch=3) == 0.0


def test_ramped_aux_weight_linearly_reaches_base_weight():
    assert Trainer._ramped_aux_weight(epoch=3, base_weight=0.5, warmup_epochs=2, start_epoch=3) == 0.25
    assert Trainer._ramped_aux_weight(epoch=4, base_weight=0.5, warmup_epochs=2, start_epoch=3) == 0.5
    assert Trainer._ramped_aux_weight(epoch=5, base_weight=0.5, warmup_epochs=2, start_epoch=3) == 0.5


def test_ramped_aux_weight_without_warmup_jumps_to_base_weight():
    assert Trainer._ramped_aux_weight(epoch=3, base_weight=0.5, warmup_epochs=0, start_epoch=3) == 0.5


def test_gate_diagnostics_extracts_scalar_means():
    trainer = Trainer.__new__(Trainer)
    outputs = {
        "gate_weights": torch.tensor([[[0.2, 0.8], [0.4, 0.6]]]),
        "q_audio": torch.tensor([0.25]),
        "q_visual": torch.tensor([0.75]),
    }
    metrics = trainer._gate_diagnostics(outputs, "train/av")
    assert metrics == {
        "train/av/gate_audio_mean": 0.30000001192092896,
        "train/av/gate_visual_mean": 0.7000000476837158,
        "train/av/q_audio_mean": 0.25,
        "train/av/q_visual_mean": 0.75,
    }


# ---- _resolve_resume_path: explicit path | auto-discover latest | none ----

def test_resolve_resume_path_explicit_exists(tmp_path):
    ckpt = tmp_path / "some.pt"
    ckpt.write_bytes(b"x")
    assert Trainer._resolve_resume_path({"resume_path": str(ckpt)}, tmp_path) == str(ckpt)


def test_resolve_resume_path_explicit_missing(tmp_path):
    assert Trainer._resolve_resume_path({"resume_path": str(tmp_path / "nope.pt")}, tmp_path) is None


def test_resolve_resume_path_auto_picks_latest_numeric(tmp_path):
    # numeric sort, not lexical: epoch_10 must beat epoch_3 (lexical would pick epoch_3).
    for e in (3, 10):
        (tmp_path / f"epoch_{e}.pt").write_bytes(b"x")
    assert Trainer._resolve_resume_path({"resume": True}, tmp_path) == str(tmp_path / "epoch_10.pt")


def test_resolve_resume_path_auto_empty_dir(tmp_path):
    # resume requested but nothing saved yet (first launch) → cold start, not a crash.
    assert Trainer._resolve_resume_path({"resume": True}, tmp_path) is None


def test_resolve_resume_path_none_when_unset(tmp_path):
    assert Trainer._resolve_resume_path({}, tmp_path) is None


# ---- save → load round-trip restores model + optimizer + scheduler + progress ----

def test_load_checkpoint_round_trip_restores_full_state(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=0)

    # Dirty the state: one real optimizer step (populates AdamW moments) + a scheduler drop.
    loss = model(torch.randn(8, 4)).pow(2).mean()
    loss.backward()
    opt.step()
    sched.step(1.0)
    sched.step(2.0)  # no improvement, patience=0 → halves lr
    saved_lr = opt.param_groups[0]["lr"]
    assert saved_lr < 1e-3  # scheduler actually changed something

    save_checkpoint(model, opt, sched, epoch=5, step=999, best_metric=0.31,
                    checkpoint_dir=str(tmp_path), filename="epoch_5.pt")

    # Fresh objects, different lr / no moment state.
    model2 = nn.Linear(4, 2)
    opt2 = optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=0)
    ckpt = load_checkpoint(str(tmp_path / "epoch_5.pt"), model2, opt2, sched2,
                           device=torch.device("cpu"))

    # progress values the trainer reads back
    assert (ckpt["epoch"], ckpt["step"], ckpt["best_metric"]) == (5, 999, 0.31)
    # weights restored
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    # optimizer restored (lr from scheduler drop + non-empty AdamW moment state)
    assert opt2.param_groups[0]["lr"] == saved_lr
    assert len(opt2.state) > 0


def test_checkpoint_round_trip_restores_warmup_scheduler_state(tmp_path):
    model = nn.Linear(4, 2)
    opt = optim.AdamW(model.parameters(), lr=5e-5)
    plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=1e-4, end_factor=1.0, total_iters=10)

    # Advance warmup once so the state is not the default constructor state.
    opt.step()
    warmup.step()
    saved_warmup_epoch = warmup.last_epoch

    save_checkpoint(
        model, opt, plateau, epoch=0, step=2, best_metric=1.0,
        checkpoint_dir=str(tmp_path), filename="epoch_0.pt", warmup_scheduler=warmup
    )

    model2 = nn.Linear(4, 2)
    opt2 = optim.AdamW(model2.parameters(), lr=5e-5)
    plateau2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min")
    warmup2 = optim.lr_scheduler.LinearLR(opt2, start_factor=1e-4, end_factor=1.0, total_iters=10)

    ckpt = load_checkpoint(
        str(tmp_path / "epoch_0.pt"), model2, opt2, plateau2,
        device=torch.device("cpu"), warmup_scheduler=warmup2
    )

    assert ckpt["warmup_scheduler_state_dict"] is not None
    assert warmup2.last_epoch == saved_warmup_epoch


def test_checkpoint_round_trip_restores_amp_scaler_state(tmp_path):
    model = nn.Linear(4, 2)
    opt = optim.AdamW(model.parameters(), lr=5e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    save_checkpoint(
        model, opt, sched, epoch=0, step=2, best_metric=1.0,
        checkpoint_dir=str(tmp_path), filename="epoch_0.pt", scaler=scaler
    )

    model2 = nn.Linear(4, 2)
    opt2 = optim.AdamW(model2.parameters(), lr=5e-5)
    sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min")
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)

    ckpt = load_checkpoint(
        str(tmp_path / "epoch_0.pt"), model2, opt2, sched2,
        device=torch.device("cpu"), scaler=scaler2
    )

    assert ckpt["scaler_state_dict"] == scaler.state_dict()
    assert scaler2.state_dict() == scaler.state_dict()


def test_resume_from_old_checkpoint_infers_completed_warmup_without_lr_jump():
    model = nn.Linear(4, 2)
    opt = optim.AdamW(model.parameters(), lr=5e-5)
    # LinearLR construction moves param-group LR to the tiny warmup start.
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=1e-4, end_factor=1.0, total_iters=500)

    # Simulate loading an old checkpoint: optimizer LR has already been restored to the fully
    # warmed value, but the checkpoint has no warmup_scheduler_state_dict.
    opt.param_groups[0]["lr"] = 5e-5

    trainer = Trainer.__new__(Trainer)
    trainer.config = {"training": {"accum_steps": 2}}
    trainer.optimizer = opt
    trainer.warmup_scheduler = warmup
    trainer.warmup_steps = 500

    trainer._sync_warmup_scheduler_after_resume({"step": 11848})

    assert warmup.last_epoch == 500
    assert opt.param_groups[0]["lr"] == 5e-5
    # This mirrors _optimizer_accum_step's warmup guard; no extra warmup step should run.
    assert not (warmup.last_epoch < trainer.warmup_steps)
