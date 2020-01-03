#!/usr/bin/env bash
source ~/.bashrc

source ./source.conf
source ./source_on.conf

export BERT_BASE_DIR=models_pretrain/albert_tiny_489k
export BERT_BASE_DIR=models_pretrain/albert_tiny_250k

export TEXT_DIR=./my_set
export MODEL_DIR=./albert_my_set_checkpoints
export PRE_MODEL_DIR=./albert_my_set_checkpoints_pre
## demo
start_finetune_step=900
start_predict_step=62

## on
start_finetune_step=8000
start_predict_step=42000

filelists=lst.pretrain.txt

function gen_ins_no_use()
{
    rm -rf $PRE_MODEL_DIR/*
    rm -rf $MODEL_DIR/*

    cd $TEXT_DIR
##    rm text.demo
##    hdxt fs -cat $hdfs_file > ./text.demo
    time python -u $gen_train_eval_ins
    cat ./train.txt | python gen_pretrain_data.py > ./pretrain.txt
    cd -
    sh -x split_file.sh $TEXT_DIR/pretrain.txt 200000 $filelists
}

function gen_ins_no_finetune()
{
    rm -rf $PRE_MODEL_DIR/*
    rm -rf $MODEL_DIR/*

    cd $TEXT_DIR
##    rm text.demo
##    hdxt fs -cat $hdfs_file > ./text.demo
    rm -rf ./pretrain.txt*
    time $python2 -u gen_pretrain_ins.py > pretrain.txt
    cd -
    sh -x split_file.sh $TEXT_DIR/pretrain.txt 200000 $filelists
}


function create_data_custom()
{

    for idx in `cat $filelists`
    do
    {
        $python create_pretraining_data.py \
            --do_whole_word_mask=True \
            --input_file=$TEXT_DIR/pretrain.txt$idx \
            --output_file=$PRE_MODEL_DIR/pre-train.tf_record.$idx \
            --vocab_file=$BERT_BASE_DIR/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=128 \
            --max_predictions_per_seq=128 \
            --masked_lm_prob=0.10
    } &
    done
    wait

}

function pretrain_custom()
{
    file_list=""
    for idx in `cat $filelists` 
    do
        file_list=$PRE_MODEL_DIR/pre-train.tf_record.$idx,$file_list
    done

    ## 如果要从头pretrain，那就注释掉init_checkpoint

    $python run_pretraining.py \
            --input_file=$file_list \
            --output_dir=$PRE_MODEL_DIR \
            --do_train=True \
            --do_eval=True \
            --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
            --train_batch_size=32 \
            --max_seq_length=128 \
            --max_predictions_per_seq=128 \
            --learning_rate=0.00176 \
            --num_train_steps=10000 \
            --num_warmup_steps=1000 \
            --save_checkpoints_steps=2000   \
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
        --num_train_epochs=1 \
        --output_dir=$MODEL_DIR \
        --init_checkpoint=$PRE_MODEL_DIR/model.ckpt-$start_finetune_step

}

function finetune_custom_from_raw()
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
        --num_train_epochs=2 \
        --output_dir=$MODEL_DIR \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

}
function predict_custom()
{
    cd $TEXT_DIR
    python $gen_predict_ins
    cd -
    # must run train in finetune first, then use its output_dir
    start_predict_step=`ls -lrt $PRE_MODEL_DIR/model.ckpt* -lrt| tail -n 1 | awk -F'model.ckpt-' '{print $2}'| awk -F'.' '{print $1}'`
    $python run_classifier.py   \
        --task_name=lcqmc_pair \
        --do_predict=true \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --output_dir=$MODEL_DIR \
        --predict_batch_size=1 \
        --init_checkpoint=$PRE_MODEL_DIR/model.ckpt-$start_predict_step

##        --init_checkpoint=$MODEL_DIR/model.ckpt-$start_predict_step
##        --init_checkpoint=$MODEL_DIR/model.ckpt-$start_predict_step
##        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 
    cd $TEXT_DIR
    python parse_res.py
}


function main()
{
    gen_ins_no_finetune
    create_data_custom
    board $MODEL_DIR 8001
    board $PRE_MODEL_DIR 8002
    pretrain_custom
#######    finetune_custom
##    finetune_custom_from_raw
    predict_custom
}

main >log/run_custom_on.log 2>&1
