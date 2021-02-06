#!/bin/bash

python3 inference_webcam.py  --model-type mattingbase --model-backbone mobilenetv2 --model-checkpoint pytorch_mobilenetv2.pth --resolution 252 168 --num-threads 2 --background green
