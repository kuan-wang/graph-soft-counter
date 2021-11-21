export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`

dataset="obqa"
model='roberta-large'
shift
shift
args=$@

elr="1e-5"
dlr="1e-2"
weight_decay="1e-3"

n_epochs=75
bs=128
mbs=4
ebs=8


k=2 #num of gnn layers
enc_dim=32

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "edge_encoder_dim $enc_dim gsc_layer $k"
echo "******************************"

save_dir_pref='saved_models'
logs_dir_pref='logs/gsc'
mkdir -p $save_dir_pref
mkdir -p $logs_dir_pref

###### Training ######
for seed in 0; do
  python3 -u gsc.py --dataset $dataset \
      --encoder $model -k $k --enc_dim $enc_dim  \
      -elr $elr -dlr $dlr -bs $bs -mbs ${mbs} -ebs ${ebs} --weight_decay ${weight_decay} --seed $seed \
      --n_epochs $n_epochs --max_epochs_before_stop 30  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --max_seq_len 88     \
      --num_relation 38    \
      --unfreeze_epoch 10 \
      --log_interval 10 \
      --save_model \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}_encdim${enc_dim}_bs${bs}__seed${seed}_${dt} $args \
  | tee -a $logs_dir_pref/train_dev_${dataset}_enc-${model}_k${k}_encdim${enc_dim}_seed${seed}_${dt}.log.txt
done
