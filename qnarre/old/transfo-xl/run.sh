#!/bin/bash

# run_enwik8_base

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_lays 12 \
        --d_model 512 \
        --n_heads 8 \
        --d_head 64 \
        --d_inner 2048 \
        --drop 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 22 \
        --multi_gpu \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi

# run_enwik8_large

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_lays 24 \
        --d_model 1024 \
        --n_heads 8 \
        --d_head 128 \
        --d_inner 3072 \
        --drop 0.15 \
        --dropatt 0.15 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 4000 \
        --max_step 400000 \
        --tgt_len 768 \
        --mem_len 768 \
        --eval_tgt_len 128 \
        --batch_size 64 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 128 \
        --mem_len 3800 \
        --clamp_len 1000 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi

# run_lm1b_base

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --adaptive \
        --n_lays 18 \
        --d_model 1024 \
        --div_val 4 \
        --n_heads 8 \
        --d_head 128 \
        --d_inner 4096 \
        --drop 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 20000 \
        --max_step 500000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 224 \
        --multi_gpu \
        --gpu0_bsz 32 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --batch_size 64 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi

# run_lm1b_large

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --adaptive \
        --div_val 4 \
        --n_lays 24 \
        --d_model 1280 \
        --n_heads 16 \
        --d_head 80 \
        --d_inner 8192 \
        --drop 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --warmup_step 30000 \
        --max_step 1200000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 512 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --batch_size 8 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi

# run_text8_base

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --n_lays 12 \
        --d_model 512 \
        --n_heads 8 \
        --d_head 64 \
        --d_inner 2048 \
        --drop 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 22 \
        --multi_gpu \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi

# run_text8_large

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --n_lays 24 \
        --d_model 1024 \
        --n_heads 8 \
        --d_head 128 \
        --d_inner 3072 \
        --drop 0.15 \
        --dropatt 0.15 \
        --optim adam \
        --lr 0.00025 \
        --tgt_len 768 \
        --mem_len 768 \
        --eval_tgt_len 128 \
        --batch_size 64 \
        --max_step 400000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 128 \
        --mem_len 3800 \
        --clamp_len 1000 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
