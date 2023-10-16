
# install requirements
pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install tensorboardX==2.4.1
pip install dgl==0.9.0
pip install fairseq==1.0.0a0+98ebe4f