from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from dd_clir.schemas.enums import EmbeddingType


class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    embedding_type: EmbeddingType = Field(
        EmbeddingType.passage, 
        description="Whether to embed a query or passage; model trained to embed differently depending"
    )
    normalized: bool = Field(False, description="Whether to normalize the embedding vector")
    model_config: ConfigDict = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "他能在多大程度上对此施加影响是很重要的，因为无论结果如何，他都将难脱干系。",
                    "embedding_type": "passage",
                    "normalize": False,
                },
                {
                    "text": "how did Alex get to the store",
                    "embedding_type": "query",
                    "normalize": True,
                }
            ]
        }
    }

    @field_validator("text", mode="after")
    def prepend_prompt(cls, text: str, info: ValidationInfo):
        prompt_start = "query: " if info.data.get("embedding_type") == EmbeddingType.query else "passage: "
        return prompt_start + text

