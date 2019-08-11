import subprocess
import cv2
import numpy as np


def from_file(filename):
    return cv2.imread(filename)


def from_adb(host='localhost:7555'):
    # subprocess.Popen(f'adb connect {host}')
    pipe = subprocess.Popen("adb shell screencap -p",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read().replace(b'\r\r\n', b'\n')
    image_bytes = np.frombuffer(image_bytes, np.uint8)
    scene = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return scene
