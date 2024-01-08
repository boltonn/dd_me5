from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Error response for the API"""

    error: bool = Field(..., example=True, title="Whether there is error")
    message: str = Field(..., example="", title="Error message")
    traceback: str = Field(None, example="", title="Detailed traceback of the error")
