#!/bin/bash

python inference_webcam.py  --model-type jit --model-backbone mobilenetv2 --model-checkpoint torchscript_mobilenetv2_fp16.pth  --resolution 720 480

# python inference_webcam.py  --model-type mattingbase --model-backbone mobilenetv2 --model-checkpoint pytorch_mobilenetv2.pth 