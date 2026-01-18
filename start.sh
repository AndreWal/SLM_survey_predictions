#!/bin/bash

docker run -p 3000:3000 -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/src:/app/src" \
  llm-survey
