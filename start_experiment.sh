clearml-task \
--project FoodDetection \
--name "FoodSeg103 Instance Swin Small" \
--docker nvcr.io/nvidia/pytorch:20.12-py3 \
--docker_bash_setup_script docker_setup_script.sh \
--docker_args="--shm-size=8g" \
--folder Swin-Transformer-Object-Detection \
--requirements requirements/build.txt \
--script tools/train.py \
--args config=configs/swin/scnet_swin_foodseg103_eren.py \
--queue default

