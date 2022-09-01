#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ldm
uvicorn main:app --proxy-headers --host 0.0.0.0 --port 8080