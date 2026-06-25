"""Back up the Modal `avsr-volume` to Google Drive via rclone, running INSIDE Modal.

Why on Modal (not `modal volume get` locally): ~1 TB through Modal's datacenter
network instead of your home uplink — the data never transits your machine.

Scope (edit BACKUP_DIRS): checkpoints + tokenizer + vicocktail_features. `vicocktail_raw`
is intentionally EXCLUDED (regenerable from the preprocessing pipeline).

Google Drive caps uploads at ~750 GB/day/account, so ~1 TB needs to span >1 day:
`--drive-stop-on-upload-limit` makes rclone exit cleanly at the cap; just RE-RUN the
same command the next day — `rclone copy` skips already-uploaded files and continues.

--- One-time setup (local) ---
1. Install rclone:                 brew install rclone        (mac)
2. Configure a Drive remote:       rclone config
     n → name it `gdrive` → storage `drive` → (client_id/secret blank for now)
     → scope `drive` → finish the browser OAuth. Creates ~/.config/rclone/rclone.conf
     with a [gdrive] section holding the refresh token.
   ⚠ For a 1 TB transfer, strongly consider creating your OWN Google OAuth client_id
     (rclone docs: "Making your own client_id") — the shared rclone client is heavily
     throttled by Google and will be slow.
3. Ship that config to Modal as a secret (single-line base64; treat as a credential —
   it contains an OAuth refresh token, do NOT commit it):
     modal secret create rclone-gdrive \
       RCLONE_CONF_B64="$(base64 < ~/.config/rclone/rclone.conf | tr -d '\n')"

--- Run ---
  modal run scripts/modal/backup_to_gdrive.py --dry-run   # test connection, transfer nothing
  modal run scripts/modal/backup_to_gdrive.py             # real copy
  # ...re-run daily until you see "ALL DIRS COMPLETE".
"""

import base64
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "avsr-backup-gdrive"
VOLUME_NAME = "avsr-volume"
SECRET_NAME = "rclone-gdrive"  # must hold key RCLONE_CONF_B64

# What to back up (top-level dirs on the volume). `vicocktail_raw` left out on purpose.
BACKUP_DIRS = ["checkpoints", "tokenizer", "vicocktail_features"]
REMOTE = "gdrive"  # rclone remote name (match your `rclone config`)
DEST_BASE = "avsr-backup"  # folder created at the root of your Drive

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "unzip", "ca-certificates")
    .run_commands("curl -fsSL https://rclone.org/install.sh | bash")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    volumes={"/mnt": volume},  # read-only in practice: rclone only reads the source
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=86400,  # 24h (Modal max). Drive's 750GB/day cap means ~1TB spans re-runs.
    cpu=4.0,
    memory=8192,  # headroom for rclone upload buffers (transfers × drive-chunk-size)
)
def backup(dry_run: bool = False, remote: str = REMOTE, dest_base: str = DEST_BASE) -> None:
    # Materialize rclone.conf from the secret.
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
        "--drive-chunk-size", "256M",
        "--tpslimit", "10",  # stay under Drive API rate limits
        "--fast-list",
        "--drive-stop-on-upload-limit",  # exit cleanly when the 750GB/day cap is hit
        "--stats", "60s", "--stats-one-line", "--verbose",
    ]
    if dry_run:
        common.append("--dry-run")

    failures = []
    for d in BACKUP_DIRS:
        src = f"/mnt/{d}"
        if not os.path.isdir(src):
            print(f"-- skip '{d}' (not found on volume) --")
            continue
        dst = f"{remote}:{dest_base}/{d}"
        print(f"\n==== rclone copy {src} -> {dst} ====")
        rc = subprocess.run(["rclone", "copy", src, dst, *common], check=False).returncode
        print(f"==== '{d}': rclone exited {rc} ====")
        if rc != 0:
            failures.append((d, rc))

    if failures:
        print("\n⚠ SOME DIRS DID NOT FINISH (most likely hit the 750GB/day Drive cap):")
        for d, rc in failures:
            print(f"   {d}: exit {rc}")
        print("Re-run the SAME command tomorrow — `rclone copy` resumes (skips done files).")
        sys.exit(1)
    print("\n✅ ALL DIRS COMPLETE")


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    backup.remote(dry_run=dry_run)
