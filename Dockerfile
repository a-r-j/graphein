FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

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


RUN mkdir -p graphein/requirements
WORKDIR /graphein


COPY .requirements /graphein/requirements
RUN echo "$(cat requirements/base.in)" >> requirements.txt \
    && echo "$(cat requirements/dev.in)" >> requirements.txt \
    && echo "$(cat requirements/extras.in)" >> requirements.txt

RUN pip install notebook==6.*
RUN pip install -r requirements.txt --no-cache-dir

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
RUN conda install -c pytorch3d pytorch3d
RUN conda install -c dglteam dgl
RUN conda install -c conda-forge ipywidgets

RUN export CUDA=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.',''))") \
    && export TORCH=$(python -c "import torch; print(torch.__version__)") \
    && pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache-dir \
    && pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache-dir \
    && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache-dir \
    && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --no-cache-dir \
    && pip install torch-geometric --no-cache-dir

RUN pip install jupyter_contrib_nbextensions
RUN jupyter nbextension enable --py widgetsnbextension

# Testing
# docker-compose -f docker-compose.cpu.yml up -d --build
# docker exec -it $(docker-compose ps -q) bash -c 'pip install -e .'
# docker exec -it $(docker-compose ps -q) bash -c 'pytest .'
# docker exec -it $(docker-compose ps -q) bash -c 'grep -l smoke_test notebooks/*.ipynb | pytest --nbval-lax --current-env'
