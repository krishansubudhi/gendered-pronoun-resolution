# Use an official TF GPU image as parent
FROM mcr.microsoft.com/azureml/bert:pretrain-openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN bash -c "source activate /opt/miniconda/envs/amlbert/ && conda install pytorch torchvision cudatoolkit=10.0 -c pytorch"
RUN bash -c "source activate /opt/miniconda/envs/amlbert/ && git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext"
RUN bash -c "source activate /opt/miniconda/envs/amlbert/ && pip install --trusted-host pypi.python.org -r requirements.txt" 
