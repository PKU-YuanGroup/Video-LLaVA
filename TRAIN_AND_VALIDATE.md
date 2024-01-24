## Data preparation

### data for training
- The images pretraining dataset is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- The images tuning dataset is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- The videos pretraining dataset is from [Valley](https://github.com/RupertLuo/Valley).
- The videos tuning dataset is from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).
- Download the training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1vPZswad5auXlDrmV7JJpdg?pwd=lj8b), [Google Disk](https://drive.google.com/file/d/1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD/view?usp=sharing) or [Peking University Disk](https://disk.pku.edu.cn:443/link/E8BFEFF8EB55E92DEEA232EB094FDB4C)


We provide the processed data on [Hugging Face](https://huggingface.co/datasets/LanguageBind/Video-LLaVA/tree/main), or you also can download from Baidu Disk as follows. 
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Disk</th>
    </tr>
    <tr align="center">
        <td>Image pretraining</td><td><a href="https://pan.baidu.com/s/17GYcE69FcJjjUM0e4Gad2w?pwd=9ga3">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Image tuning</td><td><a href="https://pan.baidu.com/s/1l-jT6t_DlN5DTklwArsqGw?pwd=o6ko">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Video pretraining</td><td><a href="https://pan.baidu.com/s/1jluOimE7mmihEBfnpwwCew?pwd=jyjz">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Video tuning</td><td><a href="https://pan.baidu.com/s/10hJ_U7wVmYTUo75YHc_n8g?pwd=g1hf">Link</a></td>
    </tr>
</table>
</div>

After downloading all of them, organize the data as follows in ```DATA_ROOT```. 

```Shell
DATA_ROOT
├── llava_image
├── llava_image_tune
├── valley
└── videochatgpt_tune
```

### data for validating
- For image, follow LLaVA's instructions. ***You MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `eval`. This also provides a general structure for all datasets.*
- For video, videos and annotations can be downloaded from Video-ChatGPT. We also provide the processed data as follows.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Disk</th><th>Google Disk</th><th>Peking University Disk</th>
    </tr>
    <tr align="center">
        <td>Activitynet_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/1d_AVx9Mz_57nA3exhQZGyA?pwd=9amr ">Link</a></td><td>-</td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>MSRVTT_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/1QHUtwHXm4Vc-Wc12XFCFsA?pwd=1rj8">Link</a></td><td><a href="https://drive.google.com/file/d/1yXh9lz7flQ5Ui2IRSd6Qi6RqSEeUJwl3/view?usp=drive_link">Link</a></td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>MSVD_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/1PJSHkjHG2BPl_ddUnBj9AA?pwd=jj34">Link</a></td><td><a href="https://drive.google.com/file/d/1_q4eiSdb7i8P3Hmh4lCfgY1uBGyzU_7X/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/8B0D01747D8AA65534820B7E60CBFEFC">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>TGIF_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/11ubtWbTtubyBmN9UPvAyow?pwd=98yr">Link</a></td><td><a href="https://drive.google.com/file/d/1so6L9rg_gdC8Segur7rKML-ffd4Ix_I6/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/B9AB387EFE8817158F181FF3D7A97163">Link</a></td>
    </tr>
</table>
</div>

After downloading all of them, organize the data as follows in `eval`.

```Shell
eval
├── GPT_Zero_Shot_QA
│   ├── Activitynet_Zero_Shot_QA
│   ├── MSRVTT_Zero_Shot_QA
│   ├── MSVD_Zero_Shot_QA
│   └── TGIF_Zero_Shot_QA
├── gqa
│   ├── answers
│   ├── data
│   └── llava_gqa_testdev_balanced.jsonl
├── llava-bench-in-the-wild
│   ├── answers
│   ├── answers_gpt4.jsonl
│   ├── bard_0718.jsonl
│   ├── bing_chat_0629.jsonl
│   ├── context.jsonl
│   ├── images
│   ├── questions.jsonl
│   ├── README.md
│   └── reviews
├── mmbench
│   ├── answers
│   ├── answers_upload
│   ├── mmbench_dev_20230712.tsv
│   └── mmbench_dev_en_20231003.tsv
├── MME
│   ├── answers
│   ├── convert_answer_to_mme.py
│   └── llava_mme.jsonl
├── mm-vet
│   ├── answers
│   ├── bard_set.json
│   ├── convert_answers.py
│   ├── images
│   ├── llava-mm-vet.jsonl
│   ├── mm-vet.json
│   └── results
├── pope
│   ├── answers
│   ├── coco
│   ├── llava_pope_test.jsonl
│   └── val2014
├── scienceqa
│   ├── answers
│   ├── images
│   ├── llava_test_CQM-A.json
│   ├── pid_splits.json
│   └── problems.json
├── seed_bench
│   ├── answers
│   ├── answers_upload
│   ├── extract_video_frames.py
│   └── llava-seed-bench.jsonl
├── textvqa
│   ├── answers
│   ├── llava_textvqa_val_v051_ocr.jsonl
│   ├── TextVQA_0.5.1_val.json
│   └── train_images
├── vizwiz
│   ├── answers
│   ├── answers_upload
│   ├── llava_test.jsonl
│   ├── test
│   ├── test.json
│   ├── train.json
│   └── val.json
└── vqav2
    ├── answers
    ├── answers_upload
    ├── llava_vqav2_mscoco_test2015.jsonl
    ├── llava_vqav2_mscoco_test-dev2015.jsonl
    └── test2015
```

## Training
Specify your `DATA_ROOT` according to the data preparation.
- Stage 1 pretraining script: [pretrain.sh](scripts/v1_5/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](scripts/v1_5/finetune.sh) or [finetune_lora.sh](scripts/v1_5/finetune_lora.sh).

## Validating
Our image validation code comes from LLaVA and our video validation code comes from Video-ChatGPT, thanks for their contribution! 

You can refer to the official repository for validation, but we also provide [off-the-shelf](scripts/v1_5/eval) scripts.

To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights. 

### MSRVTT-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_msrvtt.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_msrvtt.sh
```

### MSVD-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_msvd.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_msvd.sh
```

### TGIF-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_tgif.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_tgif.sh
```

### ActivityNet-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_activitynet.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_activitynet.sh
```


### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `eval/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/eval_image_vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `eval/vqav2/answers_upload`.

### GQA

1. Download the data following the official instructions [here](https://cs.stanford.edu/people/dorarad/gqa/download.html) and put under `eval/gqa/data`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/eval_image_gqa.sh
```

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `eval/vizwiz`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_vizwiz.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission): `eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `eval/pope`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_pope.sh
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_mmbench.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `eval/mmbench/answers_upload/mmbench_dev_20230712`.

### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `eval/llava-bench-in-the-wild`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_llavabench.sh
```

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `eval/mmvet`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_mmvet.sh
```


