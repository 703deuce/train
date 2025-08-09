# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Clone and setup ai-toolkit
RUN git clone https://github.com/ostris/ai-toolkit.git /workspace/ai-toolkit
WORKDIR /workspace/ai-toolkit

# Initialize submodules
RUN git submodule update --init --recursive

# Install ai-toolkit requirements
RUN pip install torch==2.7.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate transformers diffusers huggingface_hub

# Copy our handler and requirements
COPY requirements.txt /workspace/
COPY handler.py /workspace/
COPY dreambooth/ /workspace/dreambooth/

# Install our additional requirements and FLUX-specific requirements
RUN pip install -r /workspace/requirements.txt
RUN pip install -r /workspace/dreambooth/requirements_flux.txt

# Create necessary directories
RUN mkdir -p /workspace/datasets /workspace/outputs /workspace/configs

# Set permissions
RUN chmod +x /workspace/handler.py

# Set working directory back to workspace
WORKDIR /workspace

# Set the command to run our handler
CMD ["python", "handler.py"]