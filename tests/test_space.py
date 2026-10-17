"""The Hugging Face Space: its runtime copy matches the repository, the overlay draws, and the
uploader sends the folder to the Space and waits for the new commit to run."""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]


@pytest.mark.skipif(not (ROOT / "checkpoints" / "beat_unet" / "mps_uf_s0.pth").exists(), reason="needs the trained UNetFormer")
def test_space_runtime_matches_the_repository():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_space.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, "run python scripts/build_space.py: " + r.stdout + r.stderr


def test_draw_suggestions_marks_the_photo():
    from knot_recognition.annotate import draw_suggestions

    photo = Image.new("RGB", (400, 300), (128, 128, 128))
    out = {"lines": [[[50, 50], [150, 60], [250, 150], [350, 250]]], "ends": [[50, 50]],
           "width": {"px": 10.0, "seg": [[145, 55], [155, 65]]},
           "crossings": [{"pos": [200, 120], "o": [216, 120], "u": [200, 136], "p_over": 0.9, "under_guessed": False},
                         {"pos": [300, 200], "o": [316, 200], "u": [300, 216], "p_over": None, "under_guessed": True}]}
    im = draw_suggestions(photo, out)
    assert im.size == photo.size
    a = np.asarray(im).astype(int)
    assert (np.abs(a - 128).sum(-1) > 60).mean() > 0.01                     # something was drawn
    assert (a[..., 1] - a[..., 0] > 80).any() and (a[..., 0] - a[..., 2] > 120).any()   # green (top), orange (under)
    big = draw_suggestions(Image.new("RGB", (4000, 3000)), out, max_side=1600)
    assert max(big.size) == 1600


def test_uploader_sends_the_folder_and_waits_for_the_new_commit(monkeypatch, capsys):
    calls = {}
    stages = iter([("RUNNING", "old"), ("BUILDING", "new1"), ("RUNNING", "new1")])

    class FakeApi:
        def whoami(self):
            return {"name": "AbhyudayMishr", "orgs": []}

        def upload_folder(self, **kw):
            calls.update(kw)
            return types.SimpleNamespace(oid="new1", commit_url="https://huggingface.co/spaces/x/commit/new1")

        def get_space_runtime(self, repo):
            stage, sha = next(stages)
            return types.SimpleNamespace(stage=stage, raw={"sha": sha})

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi, fake.login = FakeApi, lambda **kw: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    import upload_space

    monkeypatch.setattr(upload_space.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["upload_space.py", "--yes", "--no-build"])
    upload_space.main()
    assert calls["repo_id"] == "AbhyudayMishr/KnotAssistant" and calls["repo_type"] == "space"
    assert Path(calls["folder_path"]) == ROOT / "huggingface_space"
    assert "*.pyc" in calls["delete_patterns"] and any("pycache" in g for g in calls["ignore_patterns"])
    out = capsys.readouterr().out
    assert "runtime/mps_uf_s0.pth" in out and "previous version" in out and "runs the new version" in out
