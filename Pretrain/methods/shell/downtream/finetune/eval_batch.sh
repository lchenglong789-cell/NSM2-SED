
source eval_env.sh

export DEVICE=1
export NPROC=2
export cmd="python /home/02363-2/SED/audiotrain/audiotrain/methods/frame/downstream/train_finetune.py"
export DEBUG=0
export n_last_blocks=1
export batch_size=90

DEBUG=1
bash eval_audioset.sh $1
exit
wait
