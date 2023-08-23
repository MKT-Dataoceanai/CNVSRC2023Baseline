import torchvision
from video_process import VideoProcess
import numpy as np
from tqdm import tqdm
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import multiprocessing

def load_video(data_filename):
    frames = torchvision.io.read_video(data_filename, pts_unit="sec")[0]
    return frames.numpy()

def extract_lip(video_process, src, vfn, dst, landmark_path):
    try:
        video = load_video(f"{src}/{vfn}")
        landmarks = np.load(landmark_path, allow_pickle=True)
        video = video_process(video, landmarks)
        Path(os.path.dirname(f"{dst}/{vfn}")).mkdir(exist_ok=True, parents=True)
        torchvision.io.write_video(f"{dst}/{vfn}", video, fps=25)
    except Exception as e:
        print(e)
        return [vfn]
    return []

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='csv files contains data info')
parser.add_argument('--src', required=True, help='path contains source video files')
parser.add_argument('--dst', required=True, help='target path of extracted video files')
parser.add_argument('--landmarks', required=True, help='path contains landmarks npy files')
parser.add_argument('--worker', default=4, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    fl_file = args.csv
    src_root = args.src
    dst_root = args.dst
    landmark_dir = args.landmarks
    worker = args.worker
    filelist = []
    with open(fl_file) as fp:
        for line in fp.readlines():
            dataset_name, vfn, frame_count, _ = line.strip('\n').split(',')
            uid = os.path.basename(vfn)[:-4]
            src = f"{src_root}/{dataset_name}"
            dst = f"{dst_root}/{dataset_name}"
            landmark = f"{landmark_dir}/{dataset_name}/{uid}.npy"
            filelist.append((vfn, src, dst, landmark))
            
    video_process = VideoProcess(convert_gray=False, window_margin=1)
    pbar = tqdm(total=len(filelist))
    pbar.set_description(f'extract lip videos')
    update = lambda *args: pbar.update()
    p = multiprocessing.Pool(processes=worker)
    res = []
    failed = []
    for vfn, src_dir, dst_dir, landmark_dir in filelist:
        res.append(p.apply_async(extract_lip, (video_process, src_dir, vfn, dst_dir, landmark_dir), callback=update))
    p.close()
    p.join()
    for r in res:
        failed.extend(r.get())
    print(f"Finish {len(filelist)} video file")
    with open(f"{fl_file.replace('/', '_')}_failed.txt", 'w') as fp:
        fp.write('\n'.join(failed))