from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


async def generic_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred."
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        Exception,
        generic_exception_handler,
    )