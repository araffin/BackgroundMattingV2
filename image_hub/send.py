import argparse
import socket

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

cam = nano.Camera(flip=0, width=width, height=height, fps=30)

while True:  # send images until Ctrl-C
    image = cam.read()
    sender.send_image(sender_name, image)
    # The execution loop will continue even if no subscriber is connected
