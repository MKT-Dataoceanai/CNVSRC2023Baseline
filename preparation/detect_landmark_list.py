import torch
import cv2
import torchvision
from detector import LandmarksDetector
from video_process import VideoProcess
import numpy as np
import random
import math
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import cProfile

def load_video(data_filename):
    return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

def extract_lip(landmarks_detector, video_process, src, vfn, dst, landmark_path):
    video = load_video(f"{src}/{vfn}.mp4")
    landmarks = landmarks_detector(f"{src}/{vfn}.mp4")
    video = video_process(video, landmarks)
    torchvision.io.write_video(f"{dst}/{vfn}.mp4", video, fps=25)
    np.save(f"{landmark_path}/{vfn}.npy", landmarks)

parser = argparse.ArgumentParser()
parser.add_argument('--list', required=True, help='path contains source video files')
parser.add_argument('--rank', default=0, type=int, help='index of current run')
parser.add_argument('--shard', default=2, type=int, help='size of multiprocessing pool')

if __name__ == '__main__':
    args = parser.parse_args()
    fl = args.list
    rank = args.rank
    shard = args.shard
    
    filelist = []
    with open(fl) as fp:
        for line in fp.readlines():
            vfn, src_dir, dst_dir, landmark_dir = line.strip().split('\t')
            filelist.append((vfn, src_dir, dst_dir, landmark_dir))
            
    landmarks_detector = LandmarksDetector(device="cuda:0")
    video_process = VideoProcess(convert_gray=False, window_margin=1)
    psize = math.ceil(len(filelist) / shard)
    filelist = filelist[rank*psize: (rank+1)*psize]
    pbar = tqdm(total=len(filelist))
    print(f"Rank {rank} init {len(filelist)} video file")
    pbar.set_description(f'rank {rank}')
    res = []
    failed = []
    for vfn, src_dir, dst_dir, landmark_dir in filelist:
        try:
            Path(dst_dir).mkdir(exist_ok=True, parents=True)
            Path(landmark_dir).mkdir(exist_ok=True, parents=True)
            extract_lip(landmarks_detector, video_process, src_dir, vfn, dst_dir, landmark_dir)
            res.append(True)
        except Exception as e:
            print(f"{vfn} failed")
            failed.append(vfn)
        pbar.update()
    print(f"Rank {rank} finish {len(filelist)} video file")
    with open(f"{fl}_failed_{rank}.txt", 'w') as fp:
        fp.write('\n'.join(failed))
