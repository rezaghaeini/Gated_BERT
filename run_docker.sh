#!/bin/bash
docker run -it --rm -v /Gated_BERT:/app/ --runtime nvidia gbert_noapex:1.0 bash
