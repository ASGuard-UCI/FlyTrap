pip install gdown
apt update && apt install unzip -y

gdown 1JWd_yHl3pBgPSuHNr-XDkllULAm5TBDK
unzip flytrap_results.zip

mkdir -p work_dirs
mv flytrap_results/* work_dirs/
rm -rf flytrap_results