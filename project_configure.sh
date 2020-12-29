# Script to build required modules

mkdir /project-configure
cd /project-configure

# apex
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd /project-configure

# tokenizers
pip install 'transformers>=3.5.1,<4.0.0'
pip install 'tokenizers==0.9.4'