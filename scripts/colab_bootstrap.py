"""Template for the single Colab cell that runs a training job.

The colab-runner agent renders this with a run id, config path, branch and commit,
and prints it for you to paste into Colab. Everything the run produces is mirrored
to Google Drive so Claude Code on your laptop can read it with tail_run.py.

Placeholders: {RUN_ID} {CONFIG} {BRANCH} {COMMIT} {REPO_URL}
"""

COLAB_CELL = r'''
# ============ BraTS-PEDs run: {RUN_ID} ============
import os, subprocess, sys, json, datetime, textwrap

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

REPO_URL   = "{REPO_URL}"
BRANCH     = "{BRANCH}"
COMMIT     = "{COMMIT}"
RUN_ID     = "{RUN_ID}"
CONFIG     = "{CONFIG}"
WORK       = "/content/brats"
MIRROR     = f"/content/drive/MyDrive/brats_runs/{RUN_ID}"

os.makedirs(MIRROR, exist_ok=True)

# --- 1. code: git is the only channel, so the run is always reproducible ---
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, WORK], check=True)
subprocess.run(["git", "-C", WORK, "fetch", "--all"], check=True)
subprocess.run(["git", "-C", WORK, "checkout", COMMIT], check=True)

# --- 2. environment header: Claude reads this to interpret timings ---
print("=" * 60, flush=True)
print("run:", RUN_ID, "| commit:", COMMIT, flush=True)
print("started:", datetime.datetime.now().isoformat(), flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
print("python:", sys.version.split()[0], flush=True)
print("=" * 60, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{WORK}/requirements.txt"], check=True)

# --- 3. train, unbuffered, tee'd straight into the Drive mirror ---
# `python -u` + tee is what makes the CLI output readable from your laptop in
# near-real-time. Without -u the log will sit in a buffer for minutes.
cmd = (
    f"cd {WORK} && python -u -m src.train "
    f"--config {CONFIG} --run-id {RUN_ID} --mirror {MIRROR} "
    f"2>&1 | tee -a {MIRROR}/train.log"
)
print("$", cmd, flush=True)
get_ipython().system(cmd)

print("FINISHED", datetime.datetime.now().isoformat(), flush=True)
'''

RESUME_CELL = r'''
# Resume {RUN_ID} after a Colab disconnect / preemption.
# Same cell as above, but src/train.py picks up last.pt from the mirror.
# Append --resume {MIRROR}/last.pt to the training command.
'''
