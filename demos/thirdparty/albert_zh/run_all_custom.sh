#!/usr/bin/env bash

source ./source.conf

export BERT_BASE_DIR=models_pretrain/albert_tiny_489k
export BERT_BASE_DIR=models_pretrain/albert_tiny_250k

export TEXT_DIR=./my_set
export MODEL_DIR=./albert_my_set_checkpoints
export PRE_MODEL_DIR=./albert_my_set_checkpoints_pre
start_finetune_step=900
start_predict_step=62
function gen_ins()
{
    rm -rf $PRE_MODEL_DIR/*
    rm -rf $MODEL_DIR/*

    cd $TEXT_DIR
    python gen_train_eval_ins.py
    cd -
}

function create_data_custom()
{
$python create_pretraining_data.py \
    --do_whole_word_mask=True \
    --input_file=$TEXT_DIR/train.txt \
    --output_file=$PRE_MODEL_DIR/pre-train.tf_record \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=128 \
    --masked_lm_prob=0.10
}


function pretrain_custom()
{
$python run_pretraining.py \
        --input_file=$PRE_MODEL_DIR/pre-train.tf_record  \
        --output_dir=$PRE_MODEL_DIR \
        --do_train=True \
        --do_eval=True \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --train_batch_size=32 \
        --max_seq_length=128 \
        --max_predictions_per_seq=128 \
        --learning_rate=0.00176 \
        --num_train_steps=1000 \
        --num_warmup_steps=100 \
        --save_checkpoints_steps=300   \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 


#        --num_train_steps=10000 \
#        --num_warmup_steps=1000 \
#        --save_checkpoints_steps=2000   \
}

function board()
{
    model=$1 
    port=$2
    ps aux|grep tensorboard |grep $port | awk '{print $2}'| xargs kill -9 
    nohup $tensorboard --logdir=./$model/ --port=$port --host=`hostname` &
}

function finetune_custom()
{
    $python run_classifier.py   \
        --task_name=lcqmc_pair   \
        --do_train=true   \
        --do_eval=true   \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --train_batch_size=64   \
        --learning_rate=1e-4 \
        --num_train_epochs=5 \
        --output_dir=$MODEL_DIR \
        --init_checkpoint=$PRE_MODEL_DIR/model.ckpt-$start_finetune_step

}

function predict_custom()
{
    export TEXT_DIR=./my_set
    cd $TEXT_DIR
    python gen_predict_ins.py
    cd -
    # must run train in finetune first, then use its output_dir
    $python run_classifier.py   \
        --task_name=lcqmc_pair \
        --do_predict=true \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --output_dir=$MODEL_DIR \
        --predict_batch_size=1 \
        --init_checkpoint=$MODEL_DIR/model.ckpt-$start_predict_step

    cd $TEXT_DIR
    python parse_res.py
}


function main()
{
    board $MODEL_DIR 8001
    board $PRE_MODEL_DIR 8002
    gen_ins
    create_data_custom
    pretrain_custom
    finetune_custom
    predict_custom
}

main >log/run_custom.log 2>&1
