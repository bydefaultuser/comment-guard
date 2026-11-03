"""Checks the boto3 Bedrock layer with botocore's Stubber, so no AWS call is made and nothing is billed."""

import dataclasses
import json

import boto3
import pytest
from botocore.stub import ANY, Stubber

from app import llm

MODEL = "example.test-model-v1:0"


def converse_reply(text: str, stop_reason="end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
        "metrics": {"latencyMs": 5},
    }


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, llm_configured=True, bedrock_model_id=MODEL))
    client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")
    judge = llm.BedrockJudge()
    judge._client = client
    with Stubber(client) as stubber:
        yield judge, stubber
        stubber.assert_no_pending_responses()


def test_parses_judgement(stubbed):
    judge, stubber = stubbed
    reply = json.dumps({"verdict": "negative_feedback", "categories": ["sarcasm"], "reason": "Sarcastic complaint."})
    stubber.add_response(
        "converse", converse_reply(reply), {"modelId": MODEL, "messages": ANY, "inferenceConfig": ANY}
    )
    result = judge.judge("Oh great, broken again.")
    assert result.status == "ok"
    assert result.model == MODEL
    assert result.judgement.verdict == "negative_feedback"
    assert result.judgement.categories == ["sarcasm"]


def test_tolerates_code_fences_and_odd_casing(stubbed):
    judge, stubber = stubbed
    reply = 'Sure!\n```json\n{"verdict": "Abusive", "categories": ["Insult", "rudeness"], "reason": "Name-calling."}\n```'
    stubber.add_response("converse", converse_reply(reply))
    result = judge.judge("you idiot")
    assert result.status == "ok"
    assert result.judgement.verdict == "abusive"
    assert result.judgement.categories == ["insult", "other"]


def test_non_json_answer_is_an_error(stubbed):
    judge, stubber = stubbed
    stubber.add_response("converse", converse_reply("I think this comment is fine."))
    assert judge.judge("hello").status == "error"


def test_content_filter_is_reported(stubbed):
    judge, stubber = stubbed
    stubber.add_response("converse", converse_reply("", stop_reason="content_filtered"))
    assert judge.judge("something nasty").status == "refused"


def test_access_denied_is_reported(stubbed):
    judge, stubber = stubbed
    stubber.add_client_error("converse", service_error_code="AccessDeniedException", service_message="no access")
    result = judge.judge("hello")
    assert result.status == "error"
    assert "credentials" in result.message


def test_comment_is_sent_inside_the_prompt(stubbed):
    judge, stubber = stubbed
    expected = {
        "modelId": MODEL,
        "messages": [{"role": "user", "content": [{"text": llm.PROMPT.format(comment="nice post")}]}],
        "inferenceConfig": {"maxTokens": 512, "temperature": 0},
    }
    stubber.add_response("converse", converse_reply('{"verdict": "clean", "categories": [], "reason": "Fine."}'), expected)
    assert judge.judge("nice post").judgement.verdict == "clean"


def test_disabled_without_model_or_credentials(monkeypatch):
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, llm_configured=False))
    assert llm.BedrockJudge().judge("hello").status == "disabled"
