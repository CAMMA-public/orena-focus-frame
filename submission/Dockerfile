FROM --platform=linux/amd64 vllm/vllm-openai:v0.21.0-x86_64-cu129-ubuntu2404 AS frame-algorithm-amd64

# Cold-start experiment: stream safetensors directly instead of mmap loading.
RUN python3 -m pip install --no-cache-dir --no-deps "instanttensor==0.1.9"
ENV INSTANTTENSOR_BACKEND=URING,AIO,URING_BUFFERED,AIO_BUFFERED,MMAP

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_DISABLE_REQUIRE=true \
    HF_HOME=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/cache \
    VLLM_CACHE_ROOT=/tmp/vllm \
    VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0

WORKDIR /opt/app

# The challenge exposes no separate model upload, so inference weights are
# baked into this image for fully offline execution.
COPY --chown=1000:1000 resources/model/ /opt/app/resources/model/
COPY --chown=1000:1000 inference.py /opt/app/

USER 1000:1000
ENTRYPOINT ["python3", "inference.py"]
