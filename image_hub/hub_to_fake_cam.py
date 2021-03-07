import cv2
import imagezmq
import pyfakewebcam
import simplejpeg

# Create fake webcam device:
# sudo modprobe v4l2loopback devices=1
# check stream:
# ffplay /dev/video2

image_hub = imagezmq.ImageHub(open_port="tcp://192.168.12.29:5555", REQ_REP=False)
fake_camera = None

while True:
    sender_name, jpg_buffer = image_hub.recv_jpg()
    image = simplejpeg.decode_jpeg(jpg_buffer, colorspace="BGR")
    if fake_camera is None:
        height, width, _ = image.shape
        fake_camera = pyfakewebcam.FakeWebcam("/dev/video2", width, height)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fake_camera.schedule_frame(image)
