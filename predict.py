import logging
import os

from pathlib import Path
import hydra
import torch
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from lightning import ModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision, torchmetrics
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from datamodule.transforms import TextTransform

from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)

video_pipeline = torch.nn.Sequential(
    FunctionalModule(lambda x: x / 255.0),
    torchvision.transforms.CenterCrop(88),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Normalize(0.421, 0.165),
)

def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

def filelist(listcsv, text_transform, cfg):
    fns = []
    lines = []
    uid2content = {}
    with open(listcsv) as fp:
        lines = fp.readlines()
    root = cfg.data.dataset.root + lines[0].split(',')[0]
    for line in lines:
        _, vfn, _, tokens = line.strip().split(',')
        uid = os.path.basename(vfn)[:-4]
        fn = f"{root}/{vfn.replace('//', '/')}"
        uid2content[uid] = text_transform.post_process([int(t) for t in tokens.split(' ')])
        if os.path.exists(fn):
            fns.append((fn,uid))
        print(fn)
    return fns, uid2content
        

@hydra.main(config_path="conf", config_name="test_cnvsrc-single")
def main(cfg):
    if not os.path.exists(cfg.data_root_dir):
        print('cfg.data_root_dir doesn\'t exist!')
        raise RuntimeError
    if not os.path.exists(cfg.code_root_dir):
        print('should set cfg.code_root_dir before running')
        raise RuntimeError
    if os.path.dirname(__file__)+'/' != cfg.code_root_dir \
        and \
        os.path.dirname(__file__) != cfg.code_root_dir:
        print('should set cfg.code_root_dir as current path')
        print(os.path.dirname(__file__))
        raise RuntimeError
    device = cfg.device
    cer = torchmetrics.CharErrorRate()
    tmpcer = torchmetrics.CharErrorRate()
    text_transform = TextTransform()
    token_list = text_transform.token_list
    model = E2E(len(token_list), cfg.model.visual_backbone).to(device)
    infer_path = cfg.infer_path
    Path(os.path.dirname(infer_path)).mkdir(exist_ok=True, parents=True)
    if cfg.infer_ckpt_path.endswith('pth'):
        model.load_state_dict(
            torch.load(cfg.infer_ckpt_path, map_location=device)
        )
    else:
        ckpt = torch.load(cfg.infer_ckpt_path, map_location=device)
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            if key[:6] == 'model.':
                state_dict[key[6:]] = value
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    fns, target_content = filelist(cfg.filelist_csv, text_transform, cfg)
    infos = {}
    beam_search = get_beam_search_decoder(model, token_list, ctc_weight=0.3,)
    for i, (fn, uid) in enumerate(fns):
        video = torchvision.io.read_video(fn, pts_unit='sec')[0] # T H W C
        video = video.permute(0, 3, 1, 2).contiguous().to(device)
        video = video_pipeline(video)
        with torch.no_grad():
            enc_feat, _ = model.encoder(video.unsqueeze(0).to(device), None)
            enc_feat = enc_feat.squeeze(0)
            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted = add_results_to_json(nbest_hyps, token_list)
            transcript = predicted.replace("▁", " ").strip().replace("<eos>", "")
        
        target = target_content[uid]
        cer.update(transcript, target)
        print(f"{uid}-pred: {transcript}")
        print(f"{uid}-targ: {target}")
        tmpcer.update(transcript, target)
        print(f'{i+1}/{len(fns)} cer:{tmpcer.compute().item()*100}%')
        print(f'{i+1}/{len(fns)} avg_cer:{cer.compute().item()*100}%')
        tmpcer.reset()
        infos[uid] = {
            'pred': transcript,
            'targ': target,
        }
    
    print(f'total-cer:{cer.errors} / {cer.total} = {cer.compute().item()*100}%')
    cer.reset()
    import json
    json.dump(infos, open(infer_path, 'w'), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
