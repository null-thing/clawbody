import json

from reachy_mini_openclaw.local_voice import parse_local_response


def test_parse_structured_response():
    response = parse_local_response(
        json.dumps(
            {
                "speech": "Hello there.",
                "actions": [
                    {"name": "look", "arguments": {"direction": "left"}}
                ],
            }
        )
    )
    assert response.speech == "Hello there."
    assert response.actions == [
        {"name": "look", "arguments": {"direction": "left"}}
    ]


def test_plain_text_is_speech_only():
    response = parse_local_response("Hello without actions.")
    assert response.speech == "Hello without actions."
    assert response.actions == []


def test_malformed_actions_are_discarded():
    response = parse_local_response(
        '{"speech":"Safe", "actions":["dance", {"name":"look","arguments":[]} ]}'
    )
    assert response.speech == "Safe"
    assert response.actions == []
