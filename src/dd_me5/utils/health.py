from enum import StrEnum

from pydantic import BaseModel


class ServiceHealthStatus(StrEnum):
    OK = "OK"
    PENDING = "PENDING"
    ERROR = "ERROR"


class ServiceHealth(BaseModel):
    status: ServiceHealthStatus = ServiceHealthStatus.PENDING


service_health = ServiceHealth()
