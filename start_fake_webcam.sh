#!/bin/bash

python inference_webcam.py  --model-type mattingbase --model-backbone mobilenetv2 --model-checkpoint pytorch_mobilenetv2.pth --resolution 180 180 --num-threads 2 --background green --fake-cam --hide-fps
