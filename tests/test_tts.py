from pathlib import Path
from subprocess import CompletedProcess

import numpy as np

from reachy_mini_openclaw.audio.tts import PiperTTS


def test_piper_uses_utf8_input_file(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        input_path = Path(command[command.index("--input-file") + 1])
        observed["text"] = input_path.read_text(encoding="utf-8")
        observed["path"] = input_path
        return CompletedProcess(command, 0, stdout=np.array([1, 2], dtype=np.int16).tobytes(), stderr=b"")

    monkeypatch.setattr("subprocess.run", fake_run)
    provider = PiperTTS("piper", "voice.onnx", 22050)
    sample_rate, audio = provider._synthesize("안녕하세요")

    assert sample_rate == 22050
    assert audio.tolist() == [[1, 2]]
    assert observed["text"] == "안녕하세요\n"
    assert not observed["path"].exists()
