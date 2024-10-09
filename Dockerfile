FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
      && \
    git lfs install --skip-smudge && \
    pip install diffusers --upgrade && \
    pip install \
        accelerate \
        boto3 \
        invisible_watermark \
        safetensors \
        transformers \
      && \
    pip cache purge && \
    mkdir /stable-diffusion /opt/artifact && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /stable-diffusion
RUN env GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 --progress \
        https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 \
        /stable-diffusion/stable-diffusion-xl-base-1.0 \
      && \
      cd stable-diffusion-xl-base-1.0 && \
      git lfs pull -I /text_encoder/model.fp16.safetensors && \
      git lfs pull -I /text_encoder_2/model.fp16.safetensors && \
      git lfs pull -I /unet/diffusion_pytorch_model.fp16.safetensors && \
      git lfs pull -I /vae/diffusion_pytorch_model.fp16.safetensors && \
      rm -rf .git

COPY runner.py /stable-diffusion/
COPY docker-entrypoint*.sh /
RUN chmod +x /docker-entrypoint*.sh /

WORKDIR /
CMD ["/bin/bash", "/docker-entrypoint.sh"]