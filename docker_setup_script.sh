apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
pip install mmcv-full
pip install terminaltables
pip install timm
pip install wandb
wandb login 126df13111573362ac4698f1eacd2e1b31e50a45
pip install clearml
pip uninstall -y pycocotools
pip install mmpycocotools
clearml-data get --id f965f0b463e74e999d11fa45f3963870
clearml-data get --id 845004a71fe84aee932c4c2efbe14799
