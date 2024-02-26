from importlib.metadata import metadata

from pydantic import BaseModel

metadata = metadata("dd_clir")


class ServiceInfo(BaseModel):
    title: str = metadata.get("Name")
    version: str = metadata.get("Version")
    description: str = metadata.get("Summary")


service_info = ServiceInfo()
