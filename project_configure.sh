# Script to build required modules

mkdir /project-configure
cd /project-configure

# apex
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd /project-configure

# tokenizers
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

git clone https://github.com/huggingface/tokenizers

cd tokenizers/bindings/python

pip install setuptools_rust
python setup.py install
