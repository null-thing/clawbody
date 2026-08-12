from reachy_mini_openclaw.config import Config


def test_local_backend_does_not_require_openai_key(monkeypatch):
    monkeypatch.setenv("VOICE_BACKEND", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Config().validate() == []


def test_openai_backend_requires_key(monkeypatch):
    monkeypatch.setenv("VOICE_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Config().validate() == [
        "OPENAI_API_KEY is required when VOICE_BACKEND=openai"
    ]


def test_openai_backend_accepts_key(monkeypatch):
    monkeypatch.setenv("VOICE_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    assert Config().validate() == []


def test_posture_diagnostic_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("POSTURE_TEST_ALERT_ON_START", raising=False)
    assert Config().POSTURE_TEST_ALERT_ON_START is False
