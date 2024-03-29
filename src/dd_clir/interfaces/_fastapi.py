from contextlib import asynccontextmanager

from dd_clir.core.models.onnx import OnnxModel
from dd_clir.schemas.request import EmbeddingRequest
from dd_clir.schemas.response import EmbeddingResponse
from dd_clir.schemas.settings import settings
from dd_clir.utils.batch import batch
from dd_clir.utils.health import ServiceHealthStatus, service_health
from dd_clir.utils.info import ServiceInfo, service_info
from dd_clir.utils.logging import logger

try:
    import uvicorn
    from fastapi import FastAPI
except ImportError:
    logger.error("Failed to import uvicorn and/or fastapi. Please install them with `pip install uvicorn fastapi`")
    exit(1)


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = OnnxModel(
        model_dir=settings.model_dir, 
        device=settings.device, 
        warmup=settings.warmup,
        quantized=settings.quantized,
    )
    service_health.status = ServiceHealthStatus.OK
    yield
    state["model"].unload()
    state.clear()


app = FastAPI(title=service_info.title, version=service_info.version, description=service_info.description, lifespan=lifespan)


@batch(max_batch_size=settings.max_batch_size, batch_wait_timeout_s=settings.batch_timeout)
async def handle_batch(batch):
    return state["model"].infer(batch=batch, normalize=False)

@app.get("/health")
async def health() -> ServiceHealthStatus:
    return service_health.status


@app.get("/info")
async def info() -> ServiceInfo:
    return service_info


@app.post("/infer", response_model_exclude_none=True)
async def infer(request: EmbeddingRequest) -> EmbeddingResponse:
    embedding = await handle_batch(request.text)
    if request.normalized:
        embedding = [e / e.norm() for e in embedding]
    return EmbeddingResponse(embedding=embedding)


def main():
    uvicorn.run("dd_clir.interfaces._api:app", host=settings.host, port=settings.port, reload=settings.reload)


if __name__ == "__main__":
    main()
