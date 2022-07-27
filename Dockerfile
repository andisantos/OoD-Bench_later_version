FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    # apt-get install --yes software-properties-common && \
    apt-get install --yes \
    python3 \
    python3-numpy \
    python3-sklearn \
    python3-pip \
    python3-matplotlib \
    python3-opencv \
    python3-tqdm \
    python3-pil \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install scipy==1.5.4 \
    Pillow==8.4.0 \
    torchvision==0.11.2 \
    torch_scatter==2.0.9 

WORKDIR /src
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
