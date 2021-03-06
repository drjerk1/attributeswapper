from videoreader import VideoReader
from signals import QuitSignal
import cv2
from yolofacedetector import FaceDetector, load_mobilenetv2_224_075_detector
from utils import rel2abs
from faceid import FACEID
import numpy as np
import argparse
from skimage import io as imageio
from attributeswapper import AttributeSwapper
from insightface import Backbone
import torch
import tensorflow as tf
import face_alignment

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

try:
    from camerawriter import CameraWriter
except Exception:
    print("V4L2 library not found, camerawriter feauture would not be avaliable")
    CameraWriter = None
    pass

quit_buttons = set([ord('q'), 27, ord('c')])

def read_image(fn):
    try:
        if fn != '':
            image = imageio.imread(fn)
            return image
    except Exception:
        print("Failed to read image")
        exit(1)
    return None

parser = argparse.ArgumentParser(description='One shot face swapper')
parser.add_argument('--input-video', type=str, dest='input_video', default='', help='Input video')
parser.add_argument('--output-video', type=str, dest='output_video', default='', help='Output video')
parser.add_argument('--output-camera', type=str, dest='output_camera', default='', help='Output camera')
parser.add_argument('--camera-width', type=int, dest='camera_width', default=1280, help='Output camera width')
parser.add_argument('--camera-height', type=int, dest='camera_height', default=720, help='Output camera height')
parser.add_argument('--target-image', type=str, dest='target_image', default='', help='Target image for face swap')
parser.add_argument('--verbose', action='store_true', dest='verbose', default=False, help='View progress in window')
parser.add_argument('--disable-faceid', dest='disable_faceid', action='store_true', default=False)

args = parser.parse_args()
output_video = args.output_video
verbose = args.verbose if output_video != '' else True
output_camera = args.output_camera
camera_width = args.camera_width
camera_height = args.camera_height
input_video = args.input_video
if input_video == '':
    input_video = 0
elif input_video.isdigit():
    input_video = int(input_video)

target_image = read_image(args.target_image)
ffmpeg_format = 'mp4v'
disable_faceid = args.disable_faceid if target_image is not None else True

def detect_single_face(image, detector, keypointdetector):
    faces = rel2abs(detector.detect(image), image)
    assert len(faces) == 1
    keypoints = keypointdetector.get_landmarks(image, detected_faces=faces)
    return faces, keypoints

def main():
    vis_debug_flag = False
    orig_debug_flag = False
    if type(input_video) is int:
        videoreader = VideoReader(input_video, width=1920, height=1080)
    else:
        videoreader = VideoReader(input_video)
    if output_video != '':
        print("Warning writing output video...")
        videowriter = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*ffmpeg_format), videoreader.get_fps(), (videoreader.get_width(), videoreader.get_height()))
    else:
        videowriter = None
    if output_camera != '':
        camerawriter = CameraWriter(output_camera, camera_width, camera_height)
    else:
        camerawriter = None

    detector = FaceDetector(load_mobilenetv2_224_075_detector("weights/facedetection-mobilenetv2-size224-alpha0.75.h5"), shots_reduce_list = [1, 3], shots_min_width_list = [1, .3])
    keypointdetector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    if not(disable_faceid):
        faceid_model = Backbone(50, 0.6, 'ir_se')
        faceid_model.load_state_dict(torch.load("weights/model_ir_se50.pth"))
        faceid = FACEID(faceid_model)
    if not(disable_faceid):
        target_faces, target_landmarks = detect_single_face(target_image, detector, keypointdetector)
        faceid.create_face_dict(["target",], [target_image,], target_faces, target_landmarks)
    attributeswapper = AttributeSwapper("weights/encoder.pth", "weights/decoder_0.pth", "weights/generator.pth")
    while True:
        try:
            frame = videoreader.read()
        except StopIteration:
            break
        vis_image = np.copy(frame)
        faces = rel2abs(detector.detect(frame), frame)

        keypoints = keypointdetector.get_landmarks(frame, detected_faces=faces)
        if not(disable_faceid):
            faceids = faceid.search_faces(frame, faces, keypoints)
        if target_image != '' and not orig_debug_flag:
            for i in range(len(faces)):
                if disable_faceid or faceids[i] == 'target':
                    kpts = keypoints[i]
                    vis_image = attributeswapper.get_image(vis_image, kpts)

        if vis_debug_flag:
            for i in range(len(faces)):
                if disable_faceid or faceids[i] == 'target':
                    color = (255,0,0)
                else:
                    color = (0,255,0)
                cv2.rectangle(vis_image, (faces[i][0], faces[i][1]), (faces[i][2], faces[i][3]), color, 1)
                for j in range(len(keypoints[i])):
                    cv2.circle(vis_image, (int(round(keypoints[i][j][0])), int(round(keypoints[i][j][1]))), 3, color)


        if videowriter is not None:
            videowriter.write(cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        if camerawriter is not None:
            camerawriter.write(vis_image)

        if verbose:
            cv2.imshow("faciallib", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            key = (cv2.waitKey(1) & 0xFF)
            if key in quit_buttons:
                if videowriter is not None:
                    videowriter.release()
                    camerawriter.close()
                raise QuitSignal(0)
            if key == ord('o'):
                orig_debug_flag = not(orig_debug_flag)
            if key == ord('v'):
                vis_debug_flag = not(vis_debug_flag)
            if key == ord('g'):
                attributeswapper.use_gan = not(attributeswapper.use_gan)
            if key == ord('s'):
                attributeswapper.use_seamless_clone = not(attributeswapper.use_seamless_clone)
            if key == ord('n'):
                attributeswapper.gen_noice()
if __name__ == "__main__":
    try:
        main()
    except QuitSignal:
        exit(0)
