#!/bin/bash

python3 inference_webcam.py  --model-type trt --model-backbone mobilenetv2 --model-checkpoint model_trt.pth --resolution 252 168 --num-threads 2 --background green
