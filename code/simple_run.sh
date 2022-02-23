#!/usr/bin/env bash

export BERT_DIR=/nfs/users/xueyou/data/bert_pretrain/electra_180g_base
export CONFIG_FILE=${BERT_DIR}/base_discriminator_config.json
export INIT_CHECKPOINT=${BERT_DIR}/electra_180g_base.ckpt
export DATA_DIR=/data/xueyou/data/corpus/task_data/LIC2019
export SEED=20190525
export OUTPUT_DIR=${DATA_DIR}/baseline
export SPATIAL_DROPOUT=0.
export EMBEDDING_DROPOUT=0.

python run_biaffine_relation.py \
  --vocab_file=vocab.txt \
  --bert_config_file=${CONFIG_FILE} \
  --init_checkpoint=${INIT_CHECKPOINT}  \
  --do_lower_case=True \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=4e-5 \
  --num_train_epochs=5.0 \
  --save_checkpoints_steps=1000 \
  --do_train=false \
  --do_eval=true \
  --use_fgm=false \
  --fgm_epsilon=0.8 \
  --fgm_loss_ratio=1.0 \
  --spatial_dropout=${SPATIAL_DROPOUT} \
  --embedding_dropout=${EMBEDDING_DROPOUT} \
  --head_lr_ratio=20.0 \
  --biaffine_size=768 \
  --electra=true \
  --amp=true \
  --seed=${SEED} \
  --data_dir=${DATA_DIR} \
  --output_dir=${OUTPUT_DIR}