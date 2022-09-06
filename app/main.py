
from fastapi import FastAPI, Request

import uuid
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

from txt2img import txt2img

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

txt2img_outdir = "/outputs/txt2img-samples"
img2img_outdir = "/outputs/img2img-samples"

os.makedirs(txt2img_outdir, exist_ok=True)
os.makedirs(img2img_outdir, exist_ok=True)

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
            {"prompt" : "a dog wearing a dress", "image" : "base64encodedimage"}
        ],
        "parameters" : {
            "ddim_steps" : 50,
            "scale" : 7.5,
            "type" : "txt2img"
        }
    }
    """
    body = await request.json()

    instances = body["instances"]
    config =  body["parameters"]
    logger.debug(f"config : {config}")

    # seed = config.get('seed',DEFAULT_SEED)
    # seed_everything(seed)

    # n_samples = config.get('n_samples', DEFAULT_N_SAMPLES)
    # batch_size = n_samples
    # n_iter = config.get('n_iter', DEFAULT_N_ITER)

    # ddim_steps = config.get('ddim_steps',DEFAULT_DDIM_STEPS)
    # scale = config.get('scale',DEFAULT_SCALE)
    # ddim_eta = config.get('ddim_eta',DEFAULT_DDIM_ETA)

    # C = config.get('C',DEFAULT_C)
    # H = config.get('H',DEFAULT_H)
    # W = config.get('W',DEFAULT_W)
    # f = config.get('f',DEFAULT_F)

    infer_type = config.get('type',"txt2img")

    if infer_type == 'txt2img':
        prompt = instances[0]["prompt"]
        retval = txt2img(model=model,
                        sampler=sampler,
                        prompt=prompt,
                        config=config,
                        outpath=txt2img_outdir)
    #elif infer_type == 'img2img':
    return {"predictions" : retval}
    # #prompt = instances[0]["prompt"]
    # data = [batch_size * [prompt]]

    # sample_path = os.path.join(outpath, "samples")
    # os.makedirs(sample_path, exist_ok=True)
    # base_count = 0

    # unique_id = str(uuid.uuid4())[:8]

    # start_code = None
    # precision_scope = autocast
    # with torch.no_grad():
    #     with precision_scope("cuda"):
    #         with model.ema_scope():
    #             tic = time.time()
    #             for n in trange(n_iter, desc="Sampling"):
    #                 for prompts in tqdm(data, desc="data"):
    #                     uc = model.get_learned_conditioning(batch_size * [""])
    #                     if isinstance(prompts,tuple):
    #                         prompts = list(prompts)
    #                     c = model.get_learned_conditioning(prompts)
    #                     shape = [C, H // f, W // f]
    #                     samples_ddim, _ = sampler.sample(S=ddim_steps,
    #                                                     conditioning=c,
    #                                                     batch_size=n_samples,
    #                                                     shape=shape,
    #                                                     verbose=False,
    #                                                     unconditional_guidance_scale=scale,
    #                                                     unconditional_conditioning=uc,
    #                                                     eta=ddim_eta,
    #                                                     x_t=start_code
    #                                                     )
    #                     x_samples_ddim = model.decode_first_stage(samples_ddim)
    #                     x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    #                     for x_sample in x_samples_ddim:
    #                         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    #                         Image.fromarray(x_sample.astype(np.uint8)).save(
    #                             os.path.join(outpath,f"{unique_id}-{base_count:05}.png"))
    #                         base_count += 1

    #             toc = time.time()

    # retval = []          
    # for i in range(base_count-1,-1,-1):
    #     img_path = os.path.join(outpath,f"{unique_id}-{i:05}.png")
    #     with open(img_path, "rb") as image_file:
    #         print("encoding image")
    #         base64_image = base64.b64encode(image_file.read())
    #         retval.append(base64_image)
    
    # return {"predictions" : retval}
