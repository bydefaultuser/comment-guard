"""End-to-end checks against the real models (the LLM layer is skipped so tests never call Bedrock)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup, which loads the models
        yield c


def check(client, text):
    response = client.post("/api/analyze", json={"text": text, "use_llm": False})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["device"] in ("cuda", "cpu")


@pytest.mark.parametrize(
    "text",
    ["Thanks, the delivery was quick and the product works great!", "The meeting is at 3pm in room 4."],
)
def test_clean_comments(client, text):
    assert check(client, text)["verdict"] == "clean"


@pytest.mark.parametrize(
    "text",
    ["This is the worst service I have ever used, totally disappointed.", "The app keeps crashing and support never replies."],
)
def test_negative_feedback(client, text):
    assert check(client, text)["verdict"] == "negative"


@pytest.mark.parametrize("text", ["You are a stupid idiot.", "What a load of shit, you moron."])
def test_abusive_comments(client, text):
    assert check(client, text)["verdict"] == "abusive"


@pytest.mark.parametrize(
    "text",
    ["this is surprisingly very good  compared to the piece of crap", "Holy shit, this is amazing! Best purchase ever."],
)
def test_swearing_in_praise_is_clean_with_profanity_flag(client, text):
    body = check(client, text)
    assert body["verdict"] == "clean"
    assert body["flags"] == ["profanity"]


def test_self_censored_swearing_is_flagged(client):
    body = check(client, "f*** yeah, finally a fix that works")
    assert body["profanity"]["words"] == ["f***"]
    assert body["verdict"] == "clean"


def test_profanity_words_are_listed_and_censored(client):
    body = check(client, "this is shit")
    assert body["profanity"]["found"] is True
    assert "shit" in body["profanity"]["words"]
    assert "shit" not in body["profanity"]["censored_text"]


def test_llm_skipped_when_turned_off(client):
    body = check(client, "hello")
    assert body["llm"]["status"] == "skipped"
    assert body["decided_by"] == "rules"


def test_empty_comment_rejected(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 422
