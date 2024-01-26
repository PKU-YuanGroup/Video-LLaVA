#!/bin/bash


CKPT_NAME="Video-LLaVA-7B"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m videollava.eval.model_vqa_science \
    --model-path ${CKPT} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 videollava/eval/eval_science_qa.py \
    --base-dir ${EVAL}/scienceqa \
    --result-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}_output.jsonl \
    --output-result ${EVAL}/scienceqa/answers/${CKPT_NAME}_result.json
