
from fastapi import FastAPI, Request

import base64
import json
import numpy as np
import pickle
import os
import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

seed_everything(42)

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sampler = PLMSSampler(model)

outdir = "/outputs/txt2img-samples"

os.makedirs(outdir, exist_ok=True)
outpath = outdir

n_samples = 3
batch_size = 3
n_rows = 3

app = FastAPI()

logging.info(f"AIP_PREDICT_ROUTE: {os.environ['AIP_PREDICT_ROUTE']}")

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    """
    Request: 
    {
        "instances" : [
            {"prompt" : "a dog wearing a dress"}
        ]
    }
    """
    body = await request.json()

    instances = body["instances"]

    prompt = instances[0]["prompt"]
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) -1

    start_code = torch.randn([3, 4, 512 // 8, 512 // 8], device=device)
    precision_scope = autocast
    all_samples = list()
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                for n in trange(2, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts,tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [4,512//8,512//8]
                        samples_ddim, _ = sampler.sample(S=50,
                                                        conditioning=c,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=7.5,
                                                        unconditional_conditioning=uc,
                                                        eta=0,
                                                        x_t=start_code
                                                        )
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        all_samples.append(x_samples_ddim)
                    
                    # save as grid
                    grid = torch.stack(all_samples,0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    retval = ""          
    grid_count -=1
    with open(os.path.join(outpath, f'grid-{grid_count:04}.png'), "rb") as image_file:
        retval = base64.b64encode(image_file.read())
    
    return {"predictions" : [retval]}
