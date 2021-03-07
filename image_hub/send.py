import argparse
import socket

import cv2
import imagezmq

import nanocamera as nano

parser = argparse.ArgumentParser(description="Inference from web-cam")

parser.add_argument(
    "--resolution",
    type=int,
    nargs=2,
    metavar=("width", "height"),
    default=(1280 // 2, 720 // 2),
)
args = parser.parse_args()

# Accept connections on all tcp addresses, port 5555
sender = imagezmq.ImageSender(connect_to="tcp://*:5555", REQ_REP=False)


sender_name = socket.gethostname()  # send RPi hostname with each image
width, height = args.resolution
jpeg_quality = 95  # 0 to 100, higher is better quality
cam = nano.Camera(flip=0, width=width, height=height, fps=30)

while True:  # send images until Ctrl-C
    image = cam.read()
    # sender.send_image(sender_name, image)
    ret_code, jpg_buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    jpg_buffer = jpg_buffer.tobytes()
    reply = sender.send_jpg(sender_name, jpg_buffer)
