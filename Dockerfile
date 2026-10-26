FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Avoid apt prompts (e.g. libcifpp-data, pulled in by dssp) hanging the build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install build-essential ffmpeg libsm6 libxext6 wget git \
    && rm -rf /var/lib/apt/lists/*

# get iputils-ping for tests
RUN apt-get update && apt-get install -y iputils-ping && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install BLAST
RUN apt-get update && apt-get install -y ncbi-blast+ && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install DSSP
RUN apt-get update && apt-get install -y dssp && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_ALWAYS_YES=true


WORKDIR /graphein

COPY pyproject.toml uv.lock ./

# Install locked dependencies into the image's Python (graphein itself is
# installed editable from the mounted repo by the docker test workflow)
RUN pip install --only-binary :all: uv==0.11.23 \
    && uv export --frozen --extra dev --extra extras --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

RUN pip install notebook==6.*

RUN conda install conda-forge::pluggy
RUN conda install -c conda-forge libgcc-ng
RUN conda install scipy scikit-learn matplotlib pandas cython ipykernel
RUN pip install ticc==0.1.4 --no-cache-dir

# # Set up vmd-python library
RUN conda install -c conda-forge vmd-python
# RUN conda install -c https://conda.anaconda.org/rbetz vmd-python

# Set up getcontacts library
RUN git clone https://github.com/getcontacts/getcontacts.git
ENV PATH /getcontacts:$PATH

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -c conda-forge ipywidgets

RUN export CUDA=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.',''))") \
    && export TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])") \
    && pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html" --no-cache-dir \
    && pip install torch-geometric==2.7.0 --no-cache-dir

RUN pip install jupyter_contrib_nbextensions
RUN jupyter nbextension enable --py widgetsnbextension

# Testing
# docker-compose -f docker-compose.cpu.yml up -d --build
# docker exec -it $(docker-compose ps -q) bash -c 'pip install -e .'
# docker exec -it $(docker-compose ps -q) bash -c 'pytest .'
# docker exec -it $(docker-compose ps -q) bash -c 'grep -l smoke_test notebooks/*.ipynb | pytest --nbval-lax --current-env'
