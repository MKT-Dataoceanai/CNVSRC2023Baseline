#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

import torchvision
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

warnings.filterwarnings("ignore")
import math
import numpy as np

class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name="resnet50"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )
        self.landmark_detector = FANPredictor(device=device, model=None)

    def __call__(self, filename):
        # import pdb
        # pdb.set_trace()
        video_frames = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()
        landmarks = []
        batch = 64
        step = math.ceil(len(video_frames)/batch)
        for i in range(step):
            video_frame_local = video_frames[i*batch:(i+1)*batch]
            if len(video_frame_local) < 1:
                break
            detected_faces_list = self.face_detector(video_frame_local, rgb=False)
            if len(detected_faces_list) != len(video_frame_local):
                detected_faces_list = self.face_detector(video_frame_local, rgb=False, old=True)
            can_batch = True # debug
            for j, detected_faces in enumerate(detected_faces_list):
                if len(detected_faces) != 1:
                    can_batch = False
                    break
            # import pdb
            # pdb.set_trace()
            if can_batch:
                face_points_list, _ = self.landmark_detector(np.array(video_frame_local), np.array(detected_faces_list), rgb=True)
                # import pdb
                # pdb.set_trace()
                for face_points in face_points_list:
                    landmarks.append(face_points)
            else:
                # print('cant batch')
                for j, detected_faces in enumerate(detected_faces_list):
                    if len(detected_faces) == 0:
                        landmarks.append(None)
                    else:
                        face_points, _ = self.landmark_detector(video_frame_local[j], detected_faces, rgb=True, old=True)
                        if face_points.shape[0] == 1:
                            landmarks.append(face_points[0])
                        else:
                            max_id, max_size = 0, 0
                            for idx, bbox in enumerate(detected_faces):
                                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                                if bbox_size > max_size:
                                    max_id, max_size = idx, bbox_size
                            landmarks.append(face_points[max_id])
        return landmarks

    # def __call__(self, filename):
    #     video_frames = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()
    #     landmarks = []
    #     for frame in video_frames:
    #         detected_faces = self.face_detector(frame, rgb=False)
    #         face_points, _ = self.landmark_detector(frame, detected_faces, rgb=True)
    #         if len(detected_faces) == 0:
    #             landmarks.append(None)
    #         else:
    #             max_id, max_size = 0, 0
    #             for idx, bbox in enumerate(detected_faces):
    #                 bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
    #                 if bbox_size > max_size:
    #                     max_id, max_size = idx, bbox_size
    #             landmarks.append(face_points[max_id])
    #     return landmarks