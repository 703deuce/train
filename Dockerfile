FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support (compatible with DeepSpeed)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install DeepSpeed with CUDA support
RUN pip install deepspeed==0.12.0

# Set working directory
WORKDIR /workspace

# Copy our handler and requirements
COPY requirements.txt /workspace/
COPY handler.py /workspace/
COPY dreambooth/ /workspace/dreambooth/

# Install our additional requirements and FLUX-specific requirements
RUN pip install -r /workspace/requirements.txt
RUN pip install -r /workspace/dreambooth/requirements_flux.txt

# Create necessary directories
RUN mkdir -p /workspace/datasets /workspace/outputs /workspace/configs /workspace/cache

# Set permissions
RUN chmod +x /workspace/handler.py

# Set the command to run our handler
CMD ["python", "handler.py"]