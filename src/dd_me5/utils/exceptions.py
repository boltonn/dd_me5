import json
import sys
import traceback

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.logger import logger
from fastapi.responses import JSONResponse


def get_error_response(exc) -> dict:
    """
    Generic error handling function
    """

    error_response = {"error": True, "message": str(exc)}

    # Add detailed traceback if in debug mode
    exc_trb, exc_type, exc_value = sys.exc_info()

    # Return traceback info if debug mode is on
    error_response["traceback"] = "".join(traceback.format_exception(exc_trb, exc_type, exc_value))

    return error_response


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handling error in validating requests
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=get_error_response(exc),
    )


async def python_exception_handler(request: Request, exc: Exception):
    """
    Handling any internal error
    """

    # Log requester infomation
    logger.error(
        "Request info:\n"
        + json.dumps(
            {
                "host": request.client.host,
                "method": request.method,
                "url": str(request.url),
                "headers": str(request.headers),
                "path_params": str(request.path_params),
                "query_params": str(request.query_params),
                "cookies": str(request.cookies),
            },
            indent=4,
        )
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=get_error_response(exc),
    )
