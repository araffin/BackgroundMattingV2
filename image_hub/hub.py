import cv2
import imagezmq
import simplejpeg

# Instantiate and provide the first publisher address
image_hub = imagezmq.ImageHub(open_port="tcp://192.168.12.29:5555", REQ_REP=False)
# image_hub.connect('tcp://192.168.86.38:5555')    # second publisher address

while True:  # show received images
    # sender_name, image = image_hub.recv_image()
    sender_name, jpg_buffer = image_hub.recv_jpg()
    image = simplejpeg.decode_jpeg(jpg_buffer, colorspace="BGR")
    cv2.imshow(sender_name, image)
    key = cv2.waitKey(1) & 0xFF
    if key in [113, 27]:
        break
