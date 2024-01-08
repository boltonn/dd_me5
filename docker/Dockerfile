FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_EXTRA_INDEX_URL https://download.pytorch.org/whl/cu11

WORKDIR /app/
COPY .. /app

# RUN addgroup --gid 1001 --system app && \
#     adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group app
# USER app

RUN pip install --no-cache-dir .[fastapi]

EXPOSE 8080
CMD uvicorn src.dd_ctranslate.interfaces._api:app --host "0.0.0.0" --port "8080" 