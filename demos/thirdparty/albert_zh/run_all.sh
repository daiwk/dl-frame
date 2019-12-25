#!/usr/bin/env bash

source ./source.conf

export BERT_BASE_DIR=models_pretrain/albert_tiny_489k
export BERT_BASE_DIR=models_pretrain/albert_tiny_250k

function create_data()
{
$python create_pretraining_data.py \
    --do_whole_word_mask=True \
    --input_file=data/news_zh_1.txt \
    --output_file=data/tf_news_2016_zh_raw_news2016zh_1.tfrecord \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=128 \
    --masked_lm_prob=0.10
}


function pretrain()
{
$python run_pretraining.py \
        --input_file=./data/tf*.tfrecord  \
        --output_dir=my_new_model_path \
        --do_train=True \
        --do_eval=True \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --train_batch_size=32 \
        --max_seq_length=128 \
        --max_predictions_per_seq=128 \
        --num_train_steps=10000 \
        --num_warmup_steps=1000 \
        --learning_rate=0.00176 \
        --save_checkpoints_steps=2000   \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 
}

function board()
{
    model=$1 ## default: my_new_model_path
    port=8003
    ps aux| grep $port | awk '{print $2}'| xargs kill -9 
    nohup $tensorboard --logdir=./$model/ --port=$port --host=`hostname` &
}

function finetune_public()
{
        export TEXT_DIR=./lcqmc # public set
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
            --output_dir=albert_lcqmc_checkpoints \
            --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

}

function finetune_custom()
{
    export TEXT_DIR=./my_set
    rm -rf albert_my_set_checkpoints/*

    cd $TEXT_DIR
    python gen_train_eval_ins.py
    cd -
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
        --output_dir=albert_my_set_checkpoints \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

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
        --output_dir=albert_my_set_checkpoints \
        --predict_batch_size=1 \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

    cd $TEXT_DIR
    python parse_res.py
}


function main()
{
    model=albert_my_set_checkpoints
    board $model
##    create_data
##    pretrain
##    finetune_public
    finetune_custom
    predict_custom
}

main >log/run.log 2>&1
