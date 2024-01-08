SHELL=/bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
NAME = dd_me5
MODEL_DIR = /home/boltonn/models/tesseract
_BUILD_ARGS_RELEASE_TAG ?= latest
_BUILD_ARGS_DOCKERFILE ?= Dockerfile

install:
	mamba env update -f environment.yml --prune

start:
	$(CONDA_ACTIVATE) $(NAME)
	conda run --no-capture-output --name $(NAME) api

test:
	$(CONDA_ACTIVATE) $(NAME)
	pytest -W ignore::DeprecationWarning

style:
	$(CONDA_ACTIVATE) $(NAME)
	black --line-length=140 .
	flake8 --max-line-length=140 . --per-file-ignores="__init__.py:F401,logging.py:F811"
	isort .

build:
	docker build -t $(NAME):$(_BUILD_ARGS_RELEASE_TAG) -f docker/$(_BUILD_ARGS_DOCKERFILE) .

run:
	docker run --env-file=.env --name $(NAME) -it --rm -v $(MODEL_DIR):/app/model -p 8080:80 $(NAME):$(_BUILD_ARGS_RELEASE_TAG)
