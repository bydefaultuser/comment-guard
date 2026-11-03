import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Blank lines like "AWS_PROFILE=" from the .env template would make boto3 look for a profile named "".
for _name in [k for k, v in os.environ.items() if k.startswith("AWS_") and not v.strip()]:
    del os.environ[_name]


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    aws_region: str
    bedrock_model_id: str
    llm_configured: bool
    force_cpu: bool
    toxicity_threshold: float
    abuse_override_threshold: float
    negative_threshold: float


def load_settings() -> Settings:
    model_id = (os.getenv("BEDROCK_MODEL_ID") or "").strip()
    # Only turn the LLM on when a model and credentials are given explicitly,
    # so a default ~/.aws profile is never billed by surprise.
    has_credentials = any(os.getenv(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK"))
    return Settings(
        aws_region=os.getenv("AWS_REGION") or "us-east-1",
        bedrock_model_id=model_id,
        llm_configured=bool(model_id) and has_credentials,
        force_cpu=_flag("FORCE_CPU"),
        toxicity_threshold=float(os.getenv("TOXICITY_THRESHOLD") or 0.5),
        abuse_override_threshold=float(os.getenv("ABUSE_OVERRIDE_THRESHOLD") or 0.7),
        negative_threshold=float(os.getenv("NEGATIVE_THRESHOLD") or 0.6),
    )


settings = load_settings()
