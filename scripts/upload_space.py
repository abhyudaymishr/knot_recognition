"""
upload_space.py - Upload huggingface_space/ to the KnotAssistant Space on Hugging Face.

Refreshes the folder from the repository first (scripts/build_space.py --skip-missing), lists what
will be uploaded, asks before uploading, and then follows the Space's rebuild until it runs (or
reports the error). Needs a Hugging Face login with write access to the Space: the first run asks
for a token (https://huggingface.co/settings/tokens, type "Write"); huggingface_hub stores it in
~/.cache/huggingface, so later runs do not ask.

  python scripts/upload_space.py                 # AbhyudayMishr/KnotAssistant
  python scripts/upload_space.py --dry-run       # only list the files
  python scripts/upload_space.py --yes           # no question before uploading
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "huggingface_space"
REPO = "AbhyudayMishr/KnotAssistant"
IGNORE = ["*__pycache__*", "*.pyc", "*.DS_Store", ".gitattributes"]
DELETE = ["*.pyc"]          # stale compiled files in the Space (they are never uploaded from here)


def local_files():
    out = []
    for p in sorted(SPACE.rglob("*")):
        rel = p.relative_to(SPACE).as_posix()
        if p.is_file() and not any(fnmatch.fnmatch(rel, g) for g in IGNORE):
            out.append((rel, p.stat().st_size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO, help="the Space, owner/name")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be uploaded")
    ap.add_argument("--yes", action="store_true", help="do not ask before uploading")
    ap.add_argument("--no-build", action="store_true", help="upload the folder as it is")
    ap.add_argument("--no-wait", action="store_true", help="do not follow the Space's rebuild")
    a = ap.parse_args()

    if not a.no_build:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_space.py"), "--skip-missing"])
        if r.returncode:
            sys.exit("the Space folder could not be refreshed (see above); nothing was uploaded")
    manifest = json.loads((SPACE / "runtime" / "MANIFEST.json").read_text())
    files = local_files()
    total = sum(s for _, s in files) / 2 ** 20
    print(f"\n{len(files)} files ({total:.1f} MB) in huggingface_space/ -> https://huggingface.co/spaces/{a.repo}")
    for rel, size in files:
        print(f"  {size / 2 ** 20:7.2f} MB  {rel}" if size > 2 ** 19 else f"  {size / 1024:7.1f} kB  {rel}")
    print(f"model {manifest['model']} ({manifest['files'][manifest['model']]['sha256']}), "
          f"strand head {manifest['strand_head']} ({manifest['files'][manifest['strand_head']]['sha256']})")
    if a.dry_run:
        return

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        sys.exit("huggingface_hub is missing: pip install huggingface_hub")
    api = HfApi()
    try:
        me = api.whoami()
    except Exception:  # noqa  (no token, or an expired one)
        print("\nNot logged in to Hugging Face. Paste a token with write access "
              "(https://huggingface.co/settings/tokens); it is stored on this Mac for later runs.")
        login(add_to_git_credential=False)
        me = api.whoami()
    owners = {me["name"]} | {o["name"] for o in me.get("orgs", [])}
    print(f"logged in as {me['name']}")
    if a.repo.split("/")[0] not in owners:
        print(f"warning: {me['name']} is not {a.repo.split('/')[0]}; the upload fails unless this account may write to the Space")

    if not a.yes:
        ans = input(f"\nUpload these files to {a.repo}? The Space rebuilds and restarts. [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit("nothing uploaded")
    msg = (f"Annotation assistant + UNetFormer parser ({manifest['model']} "
           f"{manifest['files'][manifest['model']]['sha256']}, repo {manifest.get('repository_commit') or '?'})")
    info = api.upload_folder(folder_path=str(SPACE), repo_id=a.repo, repo_type="space", commit_message=msg,
                             ignore_patterns=IGNORE, delete_patterns=DELETE)
    print(f"uploaded: {getattr(info, 'commit_url', info)}")
    if a.no_wait:
        return

    oid = getattr(info, "oid", None)
    print("following the rebuild (Ctrl-C stops watching; the Space keeps building)...")
    last, t0 = None, time.time()
    while time.time() - t0 < 20 * 60:
        try:
            rt = api.get_space_runtime(a.repo)
            stage, sha = rt.stage, (rt.raw or {}).get("sha")
        except Exception as e:  # noqa
            stage, sha = f"unknown ({type(e).__name__})", None
        new = oid is None or sha == oid               # the runtime reports the commit it runs
        shown = f"{stage}" + ("" if new else " (previous version)")
        if shown != last:
            print(f"  {time.time() - t0:5.0f} s  {shown}")
            last = shown
        if new and stage == "RUNNING":
            print(f"The Space runs the new version: https://huggingface.co/spaces/{a.repo}")
            return
        if new and stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            sys.exit(f"The Space stopped with {stage}: open https://huggingface.co/spaces/{a.repo}?logs=container "
                     "for the log")
        time.sleep(10)
    print(f"still {last} after 20 minutes: check https://huggingface.co/spaces/{a.repo}")


if __name__ == "__main__":
    main()
