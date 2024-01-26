#!/bin/bash


CKPT_NAME="Video-LLaVA-7B"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m videollava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${EVAL}/pope/val2014 \
    --answers-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 videollava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl
