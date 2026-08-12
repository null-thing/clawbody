from reachy_mini_openclaw.openclaw_bridge import PROTOCOL_VERSION, OpenClawBridge


def test_current_gateway_protocol_version():
    assert PROTOCOL_VERSION == 4


def test_gateway_url_normalization():
    assert OpenClawBridge._normalise_ws_url("http://localhost:18789") == "ws://localhost:18789"
    assert OpenClawBridge._normalise_ws_url("https://example.test") == "wss://example.test"
