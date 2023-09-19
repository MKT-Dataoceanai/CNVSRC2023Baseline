
import os
import random
import argparse

pwd_dir = os.path.dirname(os.path.abspath('.')) # preparation
filelist_dir = os.path.join(pwd_dir, 'data')

def prepare_cncvs_scp(cncvs_rootdir, dst):
    filelist = []
    for split in os.listdir(cncvs_rootdir):
        for spk in os.listdir(f"{cncvs_rootdir}/{split}"):
            if os.path.isdir(f"{cncvs_rootdir}/{split}/{spk}"):
                src_dir = os.path.join(cncvs_rootdir, split, spk, 'video')
                dst_dir = os.path.join(dst, split, spk, 'video')
                landmark_dir = os.path.join(dst, 'facelandmark')
                for fn in os.listdir(src_dir):
                    vfn = fn[:-4]
                    filelist.append(f"{vfn}\t{src_dir}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open(f'cncvs-{split}.scp', 'w') as fp:
        fp.write('\n'.join(filelist))
        
def prepare_ss_scp(single_speaker_rootdir, dst, split):
    filelist = []
    for csv in os.listdir(os.path.join(filelist_dir, 'cnvsrc-single')):
        if not csv.startswith(split):
            continue
        for line in open(os.path.join(filelist_dir, 'cnvsrc-single', csv)).readlines():
            _, path, _, _ = line.split(',')
            vfn = os.path.join(single_speaker_rootdir, os.path.dirname(path))
            dst_dir = os.path.join(dst, os.path.dirname(path))
            landmark_dir = os.path.join(dst, 'facelandmark')
            uid = os.path.basename(path).replace('.mp4', '')
            filelist.append(f"{uid}\t{vfn}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open(f'cnvsrc-single-{split}.scp', 'w') as fp:
        fp.write('\n'.join(filelist))


def prepare_ms_scp(multi_speaker_rootdir, dst, split):
    filelist = []
    for csv in os.listdir(os.path.join(filelist_dir, 'cnvsrc-multi')):
        if not csv.startswith(split):
            continue
        for line in open(os.path.join(filelist_dir, 'cnvsrc-multi', csv)).readlines():
            _, path, _, _ = line.split(',')
            vfn = os.path.join(multi_speaker_rootdir, os.path.dirname(path))
            dst_dir = os.path.join(dst, os.path.dirname(path))
            landmark_dir = os.path.join(dst, 'facelandmark')
            uid = os.path.basename(path).replace('.mp4', '')
            filelist.append(f"{uid}\t{vfn}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open(f'cnvsrc-multi-{split}.scp', 'w') as fp:
        fp.write('\n'.join(filelist))
    
parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, help='source dir of downloaded files')
parser.add_argument('--dst', required=True, help='dst dir of processed video files')
parser.add_argument('--dataset', required=True, help='which dataset to be processed', choices=['cncvs', 'cnvsrc-single', 'cnvsrc-multi'])
parser.add_argument('--split', required=True, help='train/valid/test', choices=['train', 'valid', 'test'])
if __name__== '__main__':
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    dataset = args.dataset
    split = args.split
    if dataset == 'cncvs':
        prepare_cncvs_scp(f'{src}/cncvs/', f'{dst}/cncvs/')
    elif dataset == 'cnvsrc-single':
        prepare_ss_scp(f'{src}/cnvsrc-single/', f'{dst}/cnvsrc-single/', split)
    elif dataset == 'cnvsrc-multi':
        prepare_ms_scp(f'{src}/cnvsrc-multi/', f'{dst}/cnvsrc-multi/', split)
