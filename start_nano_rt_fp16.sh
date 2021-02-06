#!/bin/bash

python3 inference_webcam.py  --model-type jit --model-backbone mobilenetv2 --model-checkpoint trt_torchscript_module_fp16.ts --resolution 252 168 --num-threads 2 --background green
