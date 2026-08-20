import sys
import threading
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
sys.modules.setdefault("soundfile", types.SimpleNamespace())


class DummyASR:
    def __init__(self):
        self.loaded = True
        self.unload_calls = 0
        self.ensure_calls = 0
        self.keep_cpu_cache_values = []

    def ensure_model_loaded(self):
        self.ensure_calls += 1
        self.loaded = True

    def unload_model(self, keep_cpu_cache=True):
        self.unload_calls += 1
        self.keep_cpu_cache_values.append(keep_cpu_cache)
        was_loaded = self.loaded
        self.loaded = False
        return {"loaded_before": was_loaded, "cpu_cached": keep_cpu_cache}

    def model_status(self):
        return {
            "loaded": self.loaded,
            "unloaded": not self.loaded,
            "cpu_cached": False,
        }


def make_engine(asr=None):
    from whisperlivekit.core import TranscriptionEngine

    TranscriptionEngine.reset()
    engine = TranscriptionEngine.__new__(TranscriptionEngine)
    engine.asr = asr
    engine.config = SimpleNamespace(
        backend="dummy",
        model_auto_unload_timeout_seconds=30.0,
        model_unload_strategy="move_to_cpu",
    )
    engine._lifecycle_lock = threading.RLock()
    engine._active_sessions = 0
    engine._auto_unload_timer = None
    return engine


def test_unload_delegates_to_backend_when_idle():
    asr = DummyASR()
    engine = make_engine(asr)

    result = engine.unload_model()

    assert result["unloaded"] is True
    assert result["loaded_before"] is True
    assert result["strategy"] == "move_to_cpu"
    assert result["cpu_cached"] is True
    assert asr.unload_calls == 1
    assert asr.keep_cpu_cache_values == [True]
    assert engine.model_status()["loaded"] is False


def test_drop_strategy_delegates_without_cpu_cache():
    asr = DummyASR()
    engine = make_engine(asr)
    engine.config.model_unload_strategy = "drop"

    result = engine.unload_model()

    assert result["strategy"] == "drop"
    assert result["cpu_cached"] is False
    assert asr.keep_cpu_cache_values == [False]


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


def test_positive_timeout_schedules_auto_unload():
    engine = make_engine(DummyASR())

    engine.register_session()
    engine.unregister_session()

    assert engine._auto_unload_timer is not None
    engine._auto_unload_timer.cancel()


def test_zero_timeout_disables_auto_unload():
    engine = make_engine(DummyASR())
    engine.config.model_auto_unload_timeout_seconds = 0.0

    engine.register_session()
    engine.unregister_session()

    assert engine._auto_unload_timer is None
