#!/bin/bash

SPLIT="mmbench_dev_20230712"

CKPT_NAME="Video-LLaVA-7B"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m videollava.eval.model_vqa_mmbench \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mmbench/$SPLIT.tsv \
    --answers-file ${EVAL}/mmbench/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME}
