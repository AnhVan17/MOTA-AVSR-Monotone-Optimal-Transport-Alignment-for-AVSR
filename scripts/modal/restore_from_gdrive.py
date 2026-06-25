"""Restore the Modal `avsr-volume` FROM a Google Drive backup via rclone, running INSIDE Modal.

Inverse of backup_to_gdrive.py: pulls gdrive:avsr-backup/<dir> back into /mnt/<dir> on the
volume. Same rclone auth (Modal secret `rclone-gdrive`).

⚠ This WRITES to the shared `avsr-volume`. Safeguards built in:
  - `rclone copy` (never `sync`) → only adds/updates, NEVER deletes anything on the volume.
  - `--update` → skips files that are NEWER on the volume (won't clobber fresher data, e.g.
    checkpoints from training that already moved past the backup). Drop it for a forced
    full overwrite from the backup.
  - Always run with `--dry-run` first to see exactly what would be written.
  - Do NOT restore over a volume that is mid-training unless you mean to.

Restore into a DIFFERENT / new volume: change VOLUME_NAME below (create it first with
`modal volume create <name>`).

Drive's download quota is far higher than the 750GB/day UPLOAD cap, so ~1TB usually fits in
fewer sessions; re-running is still safe (copy skips files already present).

Run:
  modal run scripts/modal/restore_from_gdrive.py --dry-run   # preview, write nothing
  modal run scripts/modal/restore_from_gdrive.py             # real restore
"""

import base64
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "avsr-restore-gdrive"
VOLUME_NAME = "avsr-volume"  # change to restore into a different / new volume
SECRET_NAME = "rclone-gdrive"  # same secret as the backup (key RCLONE_CONF_B64)

RESTORE_DIRS = ["checkpoints", "tokenizer", "vicocktail_features"]
REMOTE = "gdrive"
SRC_BASE = "avsr-backup"  # folder on Drive created by backup_to_gdrive.py

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "unzip", "ca-certificates")
    .run_commands("curl -fsSL https://rclone.org/install.sh | bash")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    volumes={"/mnt": volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=86400,  # 24h (Modal max)
    cpu=4.0,
    memory=8192,
)
def restore(dry_run: bool = False, remote: str = REMOTE, src_base: str = SRC_BASE) -> None:
    conf_b64 = os.environ.get("RCLONE_CONF_B64")
    if not conf_b64:
        sys.exit(f"ERROR: secret '{SECRET_NAME}' is missing key RCLONE_CONF_B64")
    conf_path = "/root/rclone.conf"
    Path(conf_path).write_text(base64.b64decode(conf_b64).decode())
    os.environ["RCLONE_CONFIG"] = conf_path

    print("== configured rclone remotes ==")
    subprocess.run(["rclone", "listremotes"], check=False)

    common = [
        "--config", conf_path,
        "--transfers", "8",
        "--checkers", "32",
        "--tpslimit", "10",  # stay under Drive API rate limits
        "--fast-list",
        "--update",  # never overwrite a file that is NEWER on the volume (drop for full overwrite)
        "--stats", "60s", "--stats-one-line", "--verbose",
    ]
    if dry_run:
        common.append("--dry-run")

    failures = []
    for d in RESTORE_DIRS:
        src = f"{remote}:{src_base}/{d}"
        dst = f"/mnt/{d}"
        print(f"\n==== rclone copy {src} -> {dst} ====")
        rc = subprocess.run(["rclone", "copy", src, dst, *common], check=False).returncode
        print(f"==== '{d}': rclone exited {rc} ====")
        if rc != 0:
            failures.append((d, rc))

    # Persist the writes so they survive the container exit.
    if not dry_run:
        volume.commit()
        print("== volume committed ==")

    if failures:
        print("\n⚠ SOME DIRS DID NOT FINISH:")
        for d, rc in failures:
            print(f"   {d}: exit {rc}")
        print("Re-run the same command — `rclone copy` resumes (skips files already present).")
        sys.exit(1)
    print("\n✅ RESTORE COMPLETE")


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    restore.remote(dry_run=dry_run)
