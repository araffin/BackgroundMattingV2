"""
Inference on webcams: Use a model on webcam input.

Once launched, the script is in background collection mode.
Press B to toggle between background capture mode and matting mode. The frame shown when B is pressed is used as background for matting.
Press Q to exit.

Example:

    python inference_webcam.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --resolution 1280 720

"""

import argparse, os, shutil, time
import cv2
import torch

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
from threading import Thread, Lock
from tqdm import tqdm
from PIL import Image

from dataset import VideoDataset
from model import MattingBase, MattingRefine

try:
    from jetcam.csi_camera import CSICamera
except ImportError:
    CSICamera = None

try:
    import pyfakewebcam

    # Create fake webcam
    # sudo modprobe v4l2loopback devices=1
except ImportError:
    pyfakewebcam = None


try:
    from torch2trt import torch2trt
except ImportError:
    torch2trt = None

try:
    import trtorch
except ImportError:
    trtorch = None

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description="Inference from web-cam")

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    choices=["mattingbase", "mattingrefine", "jit"],
)
parser.add_argument(
    "--model-backbone",
    type=str,
    required=True,
    choices=["resnet101", "resnet50", "mobilenetv2"],
)
parser.add_argument("--model-backbone-scale", type=float, default=0.25)
parser.add_argument("--model-checkpoint", type=str, required=True)
parser.add_argument(
    "--model-refine-mode",
    type=str,
    default="sampling",
    choices=["full", "sampling", "thresholding"],
)
parser.add_argument("--model-refine-sample-pixels", type=int, default=80_000)
parser.add_argument("--model-refine-threshold", type=float, default=0.7)
parser.add_argument("--num-threads", type=int, default=4)
parser.add_argument(
    "--background", type=str, choices=["white", "green", "blue", "red"], default="white"
)

parser.add_argument("--fake-cam", action="store_true")

parser.add_argument("--hide-fps", action="store_true")
parser.add_argument(
    "--resolution", type=int, nargs=2, metavar=("width", "height"), default=(1280, 720)
)
args = parser.parse_args()


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = (
            self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps
            if self._avg_fps is not None
            else fps_sample
        )
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        self.fake_cam = None
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)

    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(
                image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0)
            )
        if self.fake_cam is not None:
            image_web = np.ascontiguousarray(image, dtype=np.uint8)
            image_web = cv2.cvtColor(image_web, cv2.COLOR_RGB2BGR)
            self.fake_cam.schedule_frame(image_web)
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


if "fp16" in args.model_checkpoint:
    precision = torch.float16
else:
    precision = torch.float32


# --------------- Main ---------------

torch.set_num_threads(args.num_threads)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {args.num_threads} threads")
print(f"Using {precision} precision")
print(f"Using {device} device")

# Load model
if args.model_type == "mattingbase":
    model = MattingBase(args.model_backbone)
if args.model_type == "mattingrefine":
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
    )

if args.model_type == "jit":
    model = torch.jit.load(args.model_checkpoint)
else:
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location=device), strict=False
    )

model = model.eval().to(device=device, dtype=precision)


width, height = args.resolution

if CSICamera is None:
    cam = Camera(width=width, height=height)
else:
    cam = CSICamera(
        width=width,
        height=height,
        capture_width=1080,
        capture_height=720,
        capture_fps=30,  # reduce to reduce lag
    )
    # cam.running = True

if pyfakewebcam is not None and args.fake_cam:
    fake_cam = pyfakewebcam.FakeWebcam("/dev/video1", cam.width, cam.height)
else:
    fake_cam = None

dsp = Displayer("MattingV2", cam.width, cam.height, show_info=(not args.hide_fps))
dsp.fake_cam = fake_cam


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return (
        ToTensor()(Image.fromarray(frame))
        .unsqueeze_(0)
        .to(device=device, dtype=precision)
    )


# Convert to tensorRT
# if torch2trt is not None:
#     with torch.no_grad():
#         x = cv2_frame_to_cuda(cam.read())
#         model = torch2trt(model, [x, x])
#     # Optional: save it
#     precision_str = "fp16"
#     torch.save(model.state_dict(), f'model_trt_{precision_str}.pth')


with torch.no_grad():
    while True:
        bgr = None
        while True:  # grab bgr
            frame = cam.read()
            key = dsp.step(frame)
            if key == ord("b"):
                bgr = cv2_frame_to_cuda(cam.read())
                break
            elif key == ord("q"):
                exit()
        while True:  # matting
            frame = cam.read()

            src = cv2_frame_to_cuda(frame)

            pha, fgr = model(src, bgr)[:2]

            background_image = torch.ones_like(fgr)
            if args.background == "blue":
                background_image[:, :2, :, :] = 0
            elif args.background == "red":
                background_image[:, 1:, :, :] = 0
            elif args.background == "green":
                background_image[:, 0, :, :] = 0
                background_image[:, 2, :, :] = 0

            res = pha * fgr + (1 - pha) * background_image
            res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            key = dsp.step(res)

            if key == ord("b"):
                break
            elif key == ord("q"):
                exit()
