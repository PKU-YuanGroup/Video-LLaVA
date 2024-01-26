#!/bin/bash

CKPT_NAME="Video-LLaVA-7B"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m videollava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL}/vizwiz/test \
    --answers-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
    --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json
