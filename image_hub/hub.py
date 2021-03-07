import cv2
import imagezmq

# Instantiate and provide the first publisher address
image_hub = imagezmq.ImageHub(open_port="tcp://192.168.12.29:5555", REQ_REP=False)
# image_hub.connect('tcp://192.168.86.38:5555')    # second publisher address

while True:  # show received images
    sender_name, image = image_hub.recv_image()
    cv2.imshow(sender_name, image)  # 1 window for each unique RPi name
    key = cv2.waitKey(1) & 0xFF
    if key in [113, 27]:
        break
