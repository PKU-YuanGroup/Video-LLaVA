import argparse
import json
import os
import time

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions']

choices = ["A", "B", "C", "D"]


def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc / total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n
#     return input_ids[-len(stop_ids)]

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k: input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens


def load(ckpt_dir, model_type, cache_dir):
    # n_gpus = torch.cuda.device_count()
    n_gpus = 1

    if model_type == 'llama':
        # we use tensor parallel for loading llama
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left", cache_dir=cache_dir)

        model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16, cache_dir=cache_dir)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

    elif model_type == 'qwen':
        from videollava.model.language_model.qwen.tokenization_qwen import QWenTokenizer
        from videollava.model.language_model.qwen.modeling_qwen import QWenLMHeadModel

        model = QWenLMHeadModel.from_pretrained(ckpt_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16, cache_dir=cache_dir)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer = QWenTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left", cache_dir=cache_dir)
        tokenizer.add_special_tokens({'unk_token': '<|extra_0|>', 'bos_token': '<|extra_1|>', 'eos_token': '<|endoftext|>'})
        tokenizer.pad_token = tokenizer.unk_token

    elif model_type == 'llava':
        from videollava.mm_utils import get_model_name_from_path
        from videollava.model.builder import load_pretrained_model
        load_8bit, load_4bit = False, False
        model_base = None
        model_name = get_model_name_from_path(ckpt_dir)
        tokenizer, model, _, _ = load_pretrained_model(ckpt_dir, model_base, model_name, load_8bit, load_4bit, padding_side="left")

    model.eval()

    return model, tokenizer


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers


def main(ckpt_dir: str, param_size: str, model_type: str, cache_dir: str):
    run_results = {}
    output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)

    model, tokenizer = load(ckpt_dir, model_type, cache_dir)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({'prompt': prompt, 'answer': label})

        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers': pred_answers, 'gold_answers': gold_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='moellava/eval/mmlu_data/')
    parser.add_argument('--cache_dir', type=str, default='cache_dir')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    main(args.ckpt_dir, args.param_size, args.model_type, args.cache_dir)



'''

LLAMA_CKPT_DIR='cache_dir/models--princeton-nlp--Sheared-LLaMA-1.3B-ShareGPT'
PARAM_SIZE=1
MODEL_TYPE=llama # ["llama", "llava"] 
python3 run_mmlu_open_source.py --ckpt_dir ${LLAMA_CKPT_DIR} --param_size ${PARAM_SIZE} --model_type ${MODEL_TYPE}
'''