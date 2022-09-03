FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda

RUN git clone https://github.com/CompVis/stable-diffusion.git
WORKDIR "/stable-diffusion"
RUN ls -ll
RUN conda env create -f environment.yaml

RUN ls -ll
RUN mkdir models/ldm/stable-diffusion-v1
RUN apt-get update && apt-get install -y curl 
RUN curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > models/ldm/stable-diffusion-v1/model.ckpt




# RUN . /root/.bashrc && \
#     conda activate ldm && \
#     git clone https://github.com/TencentARC/GFPGAN.git && \
#     cd GFPGAN && pip install basicsr && pip install facexlib && \
#     pip install -r requirements.txt && python setup.py develop && \
#     pip install realesrgan
# WORKDIR "/GFPGAN"

RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN . /root/.bashrc && \
    conda activate ldm && \
    pip install uvicorn fastapi

COPY app/load_weights.py .
COPY load_weights.sh .
RUN ["./load_weights.sh"]

COPY app .
COPY entrypoint.sh .

#SHELL ["/bin/bash","-c"]
EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]

# ENTRYPOINT ["conda", "run", "-n", "ldm", \
#             "python", "scripts/txt2img.py", "--prompt", "\"55mm closeup hand photo of a breathtaking majestic beautiful armored redhead woman mage holding a tiny ball of fire in her hand on a snowy night in the village. zoom on the hand. focus on hand. dof. bokeh. art by greg rutkowski and luis royo. ultra reallistic. extremely detailed. nikon d850. cinematic postprocessing.\"","--plms"]
