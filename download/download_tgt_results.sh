pip install gdown
apt update && apt install unzip -y

gdown 1JZkyJQy3-EsUkjAC0uxpTP3LQ-_38HQm
unzip tgt_results.zip

mkdir -p work_dirs
mv tgt_results/* work_dirs/
rm -rf tgt_results