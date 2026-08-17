import threading
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
sys.modules.setdefault("soundfile", types.SimpleNamespace())

from whisperlivekit.core import TranscriptionEngine


class DummyASR:
    def __init__(self):
        self.loaded = True
        self.unload_calls = 0
        self.ensure_calls = 0

    def ensure_model_loaded(self):
        self.ensure_calls += 1
        self.loaded = True

    def unload_model(self, keep_cpu_cache=True):
        self.keep_cpu_cache = keep_cpu_cache
        self.unload_calls += 1
        was_loaded = self.loaded
        self.loaded = False
        return {"loaded_before": was_loaded, "cpu_cached": False}

    def model_status(self):
        return {
            "loaded": self.loaded,
            "unloaded": not self.loaded,
            "cpu_cached": False,
        }


def make_engine(asr=None):
    TranscriptionEngine.reset()
    engine = TranscriptionEngine.__new__(TranscriptionEngine)
    engine.asr = asr
    engine.config = SimpleNamespace(backend="dummy")
    engine._lifecycle_lock = threading.RLock()
    engine._active_sessions = 0
    return engine


def test_unload_delegates_to_backend_when_idle():
    asr = DummyASR()
    engine = make_engine(asr)

    result = engine.unload_model()

    assert result["unloaded"] is True
    assert result["loaded_before"] is True
    assert result["strategy"] == "cpu_cache"
    assert asr.keep_cpu_cache is True
    assert asr.unload_calls == 1
    assert engine.model_status()["loaded"] is False


def test_unload_refuses_while_session_active():
    asr = DummyASR()
    engine = make_engine(asr)
    engine.register_session()

    result = engine.unload_model()

    assert result == {
        "unloaded": False,
        "reason": "active_sessions",
        "active_sessions": 1,
    }
    assert asr.unload_calls == 0
    assert engine.model_status()["loaded"] is True


def test_ensure_model_loaded_delegates_to_backend():
    asr = DummyASR()
    asr.loaded = False
    engine = make_engine(asr)

    result = engine.ensure_model_loaded()

    assert result["loaded"] is True
    assert asr.ensure_calls == 1
