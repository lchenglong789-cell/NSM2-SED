cd ../../../downstream
gpu_id="2"
arch="frameamamba2"

for lr in "1e-1"
do
    echo ${arch}, learning rate: ${lr}
    python train_dcase.py \
    --nproc ${gpu_id}, \
    --learning_rate ${lr} \
    --arch ${arch} \
    --prefix _lr_${lr} \
    --pretrained_ckpt_path /data/LCL/ckpt_path/Myckpt/audiiotrain/amamba2_frame_epoch8.ckpt \
    --dcase_conf "/home/02363-2/SED/audiotrain/audiotrain/methods/frame/downstream/utils_dcase/conf/frame_40.yaml"
done