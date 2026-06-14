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
