#!/usr/bin/env bash

source ./source.conf

export BERT_BASE_DIR=models_pretrain/albert_tiny_489k
export BERT_BASE_DIR=models_pretrain/albert_tiny_250k

function board()
{
    ps aux| grep tensorboard | grep my_new_model_path| awk '{print $2}'| xargs kill -9 
    nohup $tensorboard --logdir=./my_new_model_path/ --port=8003 --host=`hostname` &
}

function predict()
{
    # must run train in lcqmc first, then use its albert_lcqmc_checkpoints
        export TEXT_DIR=./my_set
        $python run_classifier.py   \
            --task_name=lcqmc_pair \
            --do_predict=true \
            --data_dir=$TEXT_DIR   \
            --vocab_file=$BERT_BASE_DIR/vocab.txt  \
            --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
            --max_seq_length=128 \
            --output_dir=albert_lcqmc_checkpoints \
            --predict_batch_size=1 \
            --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

}

function main()
{
#    board
    predict
}

main >log/run_just_predict.log 2>&1
