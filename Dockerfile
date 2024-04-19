FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install --no-cache-dir runx==0.0.6
RUN pip install --no-cache-dir scikit-image
RUN pip install --no-cache-dir pycuda