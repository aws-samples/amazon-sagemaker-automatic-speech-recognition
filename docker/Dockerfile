# This docker file is following instructions from the official deepspeech docker.train.tmpl 
# available here https://github.com/mozilla/DeepSpeech/blob/master/Dockerfile.train.tmpl
FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:1.15.4-gpu-py37-cu100-ubuntu18.04 


ENV MOZILLA_VOICE_STT_REPO="https://github.com/mozilla/STT"
ENV MOZILLA_VOICE_STT_SHA="origin/master"

RUN apt-get update && apt-get install -y --no-install-recommends\
        apt-utils \
        bash-completion \
        build-essential \
        cmake \
        curl \
        git \
        libboost-all-dev \
        libbz2-dev \
        locales \
        python3-venv \
        unzip \
        wget
        
RUN apt-get install -y sox libsox-fmt-mp3
RUN apt-get purge -y python3-xdg

# Install dependencies for audio augmentation
RUN apt-get install -y --no-install-recommends libopus0 libsndfile1

# freeing some space
RUN rm -rf /var/lib/apt/lists/*; mkdir /opt/ml/

#Clone deepspeech library
WORKDIR /
RUN echo git clone $MOZILLA_VOICE_STT_REPO
RUN git clone $MOZILLA_VOICE_STT_REPO
RUN mv STT /opt/ml/code

#Setting the working directory of SageMaker
WORKDIR /opt/ml/code
RUN echo git checkout $MOZILLA_VOICE_STT_SHA
RUN git checkout $MOZILLA_VOICE_STT_SHA

# Build CTC decoder 
RUN cd native_client/ctcdecode && make NUM_PROCESSES=$(nproc) bindings
RUN pip3 install --upgrade native_client/ctcdecode/dist/*.whl

# Prepare deps
RUN pip3 install --upgrade pip==20.2.2 wheel==0.34.2 setuptools==49.6.0

# Install Mozilla Voice STT
#  - No need for the decoder since we did it earlier
#  - There is already correct TensorFlow GPU installed on the base image,
RUN DS_NODECODER=y DS_NOTENSORFLOW=y pip3 install --upgrade -e .

# Tool to convert output graph for inference
RUN python3 util/taskcluster.py --source tensorflow --branch r1.15 \
        --artifact convert_graphdef_memmapped_format  --target .

# Build KenLM to generate new scorers
WORKDIR /opt/ml/code/native_client
RUN rm -rf kenlm && \
        git clone https://github.com/kpu/kenlm && \
        cd kenlm && \
        git checkout 87e85e66c99ceff1fab2500a7c60c01da7315eec && \
        mkdir -p build && \
        cd build && \
        cmake .. && \
        make -j $(nproc)

RUN rm -rf /opt/ml/code/requirement* && rm -rf /opt/ml/code/setup.py

COPY start.py /opt/ml/code  
ENV PATH="/opt/ml/code:${PATH}"
WORKDIR /opt/ml/code/

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV TF_CUDNN_RESET_RND_GEN_STATE 1
ENV TF_FORCE_GPU_ALLOW_GROWTH True
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM start.py
