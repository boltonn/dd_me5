from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, conlist

from dd_me5.utils.info import service_info


class ModelAnnotation(BaseModel):
    """Generic annotation object"""

    model_name: Optional[str] = Field(service_info.title, description="Name of the model that created the annotation")
    model_version: Optional[str] = Field(service_info.version, description="Version of the model that created the annotation")

    model_config = ConfigDict(protected_namespaces=())


class MultilingualE5Response(ModelAnnotation):
    """Language annotation object"""

    embedding: conlist(float, min_length=1024, max_length=1024) = Field(..., description="Language embedding")