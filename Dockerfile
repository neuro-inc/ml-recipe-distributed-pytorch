FROM neuromation/base:v1.7.6

RUN mkdir /project
WORKDIR /project

COPY apt.txt .
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    cat apt.txt | tr -d "\r" | xargs -I % apt-get -qq install --no-install-recommends % && \
    apt-get -qq clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    pip install --no-cache-dir 'transformers>=3.5.1,<4.0.0' 'tokenizers==0.9.4'

COPY setup.cfg .

COPY requirements.txt .
RUN pip install --progress-bar=off -U --no-cache-dir -r requirements.txt

RUN ssh-keygen -f /id_rsa -t rsa -N neuromation -q