from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, conlist

from dd_clir.utils.info import service_info


class ModelAnnotation(BaseModel):
    """Generic annotation object"""

    model_name: Optional[str] = Field(service_info.title, description="Name of the model that created the annotation")
    model_version: Optional[str] = Field(service_info.version, description="Version of the model that created the annotation")

    model_config = ConfigDict(protected_namespaces=())


class EmbeddingResponse(ModelAnnotation):
    """Language annotation object"""

    embedding: list[float] = Field(..., description="Language embedding")