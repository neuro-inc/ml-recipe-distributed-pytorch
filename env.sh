#export RPOJECT_PATH=/project-env
mkdir /project-env
cd /project-env

# apex
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd /project-env

# tokenizers
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

git clone https://github.com/huggingface/tokenizers

cp -v /qa-competition/config/byte_level_bpe.py ./tokenizers/bindings/python/tokenizers/implementations/

cd tokenizers/bindings/python

pip install setuptools_rust
python setup.py install
