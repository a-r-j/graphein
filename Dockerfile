FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update \
    && apt-get -y install build-essential ffmpeg libsm6 libxext6 wget git \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_ALWAYS_YES=true

WORKDIR /tmp
RUN git clone https://www.github.com/a-r-j/graphein

WORKDIR /tmp/graphein
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "graphein", "/bin/bash", "-c"]
RUN pip install -e .
RUN python -m ipykernel install --user --name=graphein

RUN conda install -c conda-forge libgcc-ng
RUN conda install scipy scikit-learn matplotlib pandas cython ipykernel
RUN pip install ticc==0.1.4

# Set up vmd-python library
RUN conda install -c conda-forge vmd-python
# RUN conda install -c https://conda.anaconda.org/rbetz vmd-python

# Set up getcontacts library
RUN git clone https://github.com/getcontacts/getcontacts.git
ENV PATH /getcontacts:$PATH

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# RUN conda install -c pytorch pytorch
RUN conda install -c pytorch3d pytorch3d
RUN conda install -c dglteam dgl

RUN export TORCH=1.7.1 \
    && export CUDA=cu110 \
    && pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    && pip install torch-geometric

RUN pip install --upgrade numpy scipy pandas

RUN mkdir -p /opt/notebooks

EXPOSE 8888
ENTRYPOINT ["conda", "run", "-n", "graphein", "jupyter", "notebook", "--notebook-dir=/opt/notebooks",  "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''", "--port=8888",  "--no-browser", "--allow-root"]

