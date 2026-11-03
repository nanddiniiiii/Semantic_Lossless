"""Recovery cases that matter when a chunk collector is interrupted."""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from src.person1 import collection
from src.person1.collection import preserve_chunks
from src.sharing import MODEL_NAMES


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_empty_owned_inventory_is_confirmed_absence(monkeypatch):
    monkeypatch.setattr(collection.subprocess,'run',lambda *a,**k:
                        types.SimpleNamespace(returncode=0,stdout='No datasets found\n',stderr=''))
    assert collection.owned_dataset_exists('owner/p1-checkpoint',{}) is False


def test_existing_partial_directory_recovers_missing_checkpoint(tmp_path):
    """A crash after mkdir must not strand the only continuation checkpoint."""
    root = tmp_path / "repo"
    destination = tmp_path / "retrieved"
    run_id = "chunk-one"
    run = destination / "published_output" / "session" / "deepjscc"
    checkpoint = run / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"real checkpoint bytes")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _write(destination / "submitted_version.json", {"kernel": "owner/semantic-notebook-1"})
    _write(run / "chunk_checkpoint.json", {
        "algorithm": "deepjscc", "checkpoint": "checkpoints/last.pt", "sha256": sha,
    })
    _write(run / "person1_run.json", {"run_id": run_id})
    _write(run / "training_summary.json", {"contract": "fixed"})
    _write(destination / "checkpoint_dataset-deepjscc.json", {
        "dataset": "owner/private-checkpoint", "status": "ready", "sha256": sha,
    })
    target = root / "Models" / "teammate_1_models" / MODEL_NAMES["deepjscc"] / run_id / "partial"
    target.mkdir(parents=True)  # prior attempt crashed here
    preserve_chunks(root, destination, {})
    assert (target / "last.pt").read_bytes() == checkpoint.read_bytes()


@pytest.mark.parametrize('algorithm,filename',[('deepjscc','last.pt'),('adjscc','last-tf.zip')])
def test_upload_started_with_remote_absent_retries_private_creation(tmp_path, monkeypatch,algorithm,filename):
    """An interrupted upload receipt must not suppress the next create attempt."""
    root = tmp_path / "repo"
    destination = tmp_path / "retrieved"
    run = destination / "published_output" / "session" / algorithm
    checkpoint = run / "checkpoints" / filename
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint for retry")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    run_id = "chunk-retry"
    _write(destination / "submitted_version.json", {"kernel": "owner/semantic-notebook-1"})
    _write(run / "chunk_checkpoint.json", {
        "algorithm": algorithm, "checkpoint": "checkpoints/"+filename, "sha256": sha,
    })
    _write(run / "person1_run.json", {"run_id": run_id})
    _write(run / "training_summary.json", {"contract": "fixed"})
    receipt = destination / ('checkpoint_dataset-'+algorithm+'.json')
    _write(receipt, {"dataset": "owner/p1-deepjscc-stale", "status": "upload_started", "sha256": sha})

    monkeypatch.setattr(collection, "owned_dataset_exists", lambda ref, env: False)
    calls = []

    def fake_run(argv, **kwargs):
        assert kwargs.get('encoding')=='utf-8'
        calls.append(argv)
        if argv[1:3] == ["datasets", "create"]:
            return types.SimpleNamespace(returncode=0, stdout="created \u2713 \U0001f680", stderr=None)
        if argv[1:3] == ["datasets", "status"]:
            return types.SimpleNamespace(returncode=0, stdout="ready", stderr="")
        pytest.fail("unexpected Kaggle command")

    monkeypatch.setattr(collection.subprocess, "run", fake_run)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def datasets(self):
            return types.SimpleNamespace(dataset_api_client=types.SimpleNamespace(
                get_dataset_metadata=lambda request: types.SimpleNamespace(
                    info=types.SimpleNamespace(is_private=True))))

    class FakeApi:
        def authenticate(self):
            pass

        def build_kaggle_client(self):
            return FakeClient()

    packages = ["kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended",
                "kagglesdk", "kagglesdk.datasets", "kagglesdk.datasets.types",
                "kagglesdk.datasets.types.dataset_api_service"]
    for name in packages:
        module = types.ModuleType(name)
        if name in ("kaggle", "kaggle.api", "kagglesdk", "kagglesdk.datasets", "kagglesdk.datasets.types"):
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = FakeApi
    sys.modules["kagglesdk.datasets.types.dataset_api_service"].ApiGetDatasetMetadataRequest = type(
        "ApiGetDatasetMetadataRequest", (), {})

    preserve_chunks(root, destination, {})
    assert len([argv for argv in calls if argv[1:3] == ["datasets", "create"]]) == 1
    assert json.loads(receipt.read_text())["status"] == "ready"
    assert json.loads(receipt.read_text())["is_private"] is True
    assert '\U0001f680' in (destination/('checkpoint_upload-'+algorithm+'.log')).read_text(encoding='utf-8')
    upload=Path(next(argv[argv.index('-p')+1] for argv in calls if argv[1:3]==['datasets','create']))
    payload=json.loads((upload/'resume_receipt.json').read_text())['checkpoint']
    assert (upload/payload).read_bytes()==checkpoint.read_bytes()
    if filename.endswith('.zip'):
        assert payload=='checkpoint.bin'
        assert not list(upload.glob('*.zip'))
