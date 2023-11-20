

<p align="center">
    <img src="https://z1.ax1x.com/2023/11/07/pil4sqH.png" width="150" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2311.10122">Video-LLaVA: Learning United Visual Representation by Alignment Before Projection</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>



<h5 align="center">
    
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Video-LLaVA)
[![arXiv](https://img.shields.io/badge/Arxiv-2311.10122-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.10122)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPKU-YuanGroup%2FVideo-LLaVA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/PKU-YuanGroup/Video-LLaVA?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/PKU-YuanGroup/Video-LLaVA?color=success&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aissue+is%3Aclosed)

</h5>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-llava-learning-united-visual/zeroshot-video-question-answer-on-msrvtt-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msrvtt-qa?p=video-llava-learning-united-visual) <br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-llava-learning-united-visual/zeroshot-video-question-answer-on-msvd-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msvd-qa?p=video-llava-learning-united-visual) <br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-llava-learning-united-visual/zeroshot-video-question-answer-on-tgif-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-tgif-qa?p=video-llava-learning-united-visual) <br>
<!--[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-llava-learning-united-visual/zeroshot-video-question-answer-on-activitynet)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-activitynet?p=one-for-all-video-conversation-is-feasible) <br>-->


## 📰 News
* **[2023.11.20]**  🤗[Demo](https://huggingface.co/spaces/LanguageBind/Video-LLaVA) and code are available now! Welcome to **watch** 👀 this repository for the latest updates.

## 😮 Highlights

Video-LLaVA exhibits remarkable interactive capabilities between images and videos, despite the absence of image-video pairs in the dataset.

### 💡 Simple baseline, learning united visual representation by alignment before projection
- With **the binding of unified visual representations to the language feature space**, we enable an LLM to perform visual reasoning capabilities on both images and videos simultaneously.

### 🔥 High performance, complementary learning with video and image
- Extensive experiments demonstrate **the complementarity of modalities**, showcasing significant superiority when compared to models specifically designed for either images or videos. 

<img src="assets/main.jpg"/>

## 🤗 Demo

* **Gradio Web UI**

Highly recommend trying out our web demo by the following command, which incorporates all features currently supported by Video-LLaVA. We also provide [online demo](https://huggingface.co/spaces/LanguageBind/Video-LLaVA) in Huggingface Spaces.
```bash
python -m  llava.serve.gradio_web_server
```


https://github.com/PKU-YuanGroup/Video-LLaVA/assets/62638829/71ab15ac-105e-4b18-b0b5-e1b35d70607b



* **CLI Inference**

```bash
python -m llava.serve.cli --model-path "LanguageBind/Video-LLaVA-7B" --image-file "path/to/your/image.jpg" --load-4bit
```

<img src="assets/videocli.gif" width="500" />

```bash
python -m llava.serve.cli --model-path "LanguageBind/Video-LLaVA-7B" --video-file "path/to/your/video.mp4" --load-4bit
```

<img src="assets/imagecli.gif" width="500" />

## 🚀 Main Results

### Image understanding
<p align="left">
<img src="assets/res_img.jpg" width=80%>
</p>

### Video understanding
<p align="left">
<img src="assets/res_vi.jpg" width=80%>
</p>

## 🛠️ Requirements and Installation
* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/Video-LLaVA
cd Video-LLaVA
conda create -n videollava python=3.10 -y
conda activate videollava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

## 🗝️ Training & Validating
The training & validating instruction is in [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md).

## 👍 Acknowledgement
* [LLaVA](https://github.com/haotian-liu/LLaVA) The codebase we built upon and it is an efficient large language and vision assistant.
* [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) Great job contributing the evaluation code and dataset.

## 🤝 Related Projects
* [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) An open source language-based retrieval framework.

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/LICENSE) file.
* The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@misc{lin2023videollava,
      title={Video-LLaVA: Learning United Visual Representation by Alignment Before Projection}, 
      author={Bin Lin and Bin Zhu and Yang Ye and Munan Ning and Peng Jin and Li Yuan},
      year={2023},
      eprint={2311.10122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


<!--
## ✨ Star History
[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/Video-LLaVA&type=Date)](https://star-history.com/#PKU-YuanGroup/Video-LLaVA&Date)
-->
