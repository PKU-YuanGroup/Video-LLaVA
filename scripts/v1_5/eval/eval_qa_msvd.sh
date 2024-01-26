

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
output_name="Video-LLaVA-7B"
pred_path="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/gpt3.5"
output_json="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/results.json"
api_key=""
api_base=""
num_tasks=8



python3 videollava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}