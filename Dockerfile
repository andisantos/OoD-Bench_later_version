FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime AS base

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src

RUN apt-get update && \
    apt-get install --yes software-properties-common && \
    apt-get install --yes \
    python3 \
    python3-numpy \
    python3-pip \
    python3-matplotlib \
    python3-opencv \
    python3-tqdm \
    python3-pil \
    vim \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install scipy==1.5.4 \
    scikit-learn==1.0.2 \
    Pillow==8.4.0 \
    torchvision==0.11.2 \
    torch_scatter==2.0.9 \
    pandas==1.3.5 

CMD ["/bin/bash"]

FROM base AS base-jupyter

RUN apt-get update && \
    apt-get install --yes \
    python3-seaborn \
    jupyter && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", \
            "--no-browser", "--allow-root", \
            "--notebook-dir=/notebooks"]
