"""Jev-compatible Pydantic v2 schemas for the SystemOne API.

Mirrors ``POST /v1/systemone`` from the TypeSafe docs:

* Request: ``{state, model, questions}`` where each question is one of
  choice / score / noul.
* Response: ``{model, answers, usage}`` with typed answers keyed by the
  caller-chosen question ids.

Wire format notes:

* Score ``probabilities`` / ``legend`` keys are strings on the wire
  (``{"0": ...}``); the Python SDK exposes them as ints. This server
  speaks the HTTP wire format.
* ``instructions`` and criteria descriptions accept ``str | dict | list``
  (structured prompts); dict/list values are rendered by the backend.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# JSON scalar structures accepted anywhere Jev allows "structured" text.
StructuredText = Union[str, dict[str, Any], list[Any]]

# Raw JSON state: plain text, a record/object, or a sequence of records.
StateT = Union[str, dict[str, Any], list[Any]]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: StructuredText | None = Field(
        default=None, description="What a yes (value near 1) means."
    )
    false: StructuredText | None = Field(
        default=None, description="What a no (value near 0) means."
    )


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: StructuredText
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: StructuredText
    criteria: Annotated[
        dict[str, StructuredText | None],
        Field(min_length=1, max_length=MAX_CHOICE_OPTIONS),
    ]


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: StructuredText
    criteria: Annotated[
        list[StructuredText],
        Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS),
    ]


QuestionT = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    """``POST /v1/systemone`` request body."""

    model_config = ConfigDict(extra="forbid")

    state: StateT
    model: str = Field(min_length=1)
    questions: Annotated[dict[str, QuestionT], Field(min_length=1)]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    noul: Annotated[float, Field(ge=0.0, le=1.0)]


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    score: float
    legend: dict[str, StructuredText]
    probabilities: dict[str, float]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


AnswerT = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class SystemOneResponse(BaseModel):
    """``POST /v1/systemone`` response body."""

    model_config = ConfigDict(extra="forbid")

    model: str
    answers: dict[str, AnswerT]
    usage: Usage


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    release_date: str


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelCard]


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
