from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextPart(StrictModel):
    type: Literal["text"]
    text: str = Field(max_length=262144)


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]

    def plain(self):
        text = self.content if isinstance(self.content, str) else "".join(p.text for p in self.content)
        if len(text) > 262144:
            raise ValueError("Message is too long")
        return {"role": "system" if self.role == "developer" else self.role, "content": text}


Stop = Annotated[str, Field(min_length=1, max_length=256)]


class StreamOptions(StrictModel):
    include_usage: bool = False


class Common(StrictModel):
    model: str = Field(min_length=1, max_length=512)
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    stream: bool = False


class ChatRequest(Common):
    messages: list[Message] = Field(min_length=1, max_length=256)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    stop: Stop | list[Stop] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 32769)
    n: Literal[1] = 1
    stream_options: StreamOptions | None = None
    user: str | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if "max_tokens" in self.model_fields_set and self.max_completion_tokens is not None:
            raise ValueError("Use max_tokens or max_completion_tokens, not both")
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("At most four stop sequences are supported")
        return self

    def generation(self):
        return {
            "model": self.model,
            "messages": [m.plain() for m in self.messages],
            "max_tokens": self.max_completion_tokens or self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": [self.stop] if isinstance(self.stop, str) else self.stop or [],
        }


class CompletionRequest(Common):
    prompt: str = Field(max_length=262144)
    stop: Stop | list[Stop] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 32769)
    n: Literal[1] = 1
    stream_options: StreamOptions | None = None
    user: str | None = None

    def generation(self):
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("At most four stop sequences are supported")
        return {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": [self.stop] if isinstance(self.stop, str) else self.stop or [],
        }


class AnthropicRequest(Common):
    max_tokens: int = Field(ge=1, le=32768)
    messages: list[Message] = Field(min_length=1, max_length=256)
    system: str | list[TextPart] | None = None
    stop_sequences: list[Stop] = Field(default_factory=list, max_length=4)
    temperature: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_roles(self):
        if any(m.role not in ("user", "assistant") for m in self.messages):
            raise ValueError("Anthropic messages accept user/assistant roles; use the top-level system field")
        return self

    def generation(self):
        messages = [m.plain() for m in self.messages]
        if self.system:
            content = self.system if isinstance(self.system, str) else "".join(p.text for p in self.system)
            messages.insert(0, {"role": "system", "content": content})
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_sequences,
        }


class LoadRequest(StrictModel):
    model: str = Field(min_length=1, max_length=512)
