from modal import Volume, Image, Stub, Mount, Secret, build
from pathlib import Path
REPO_HOME = "/app"
VOLUME_DIR = "/volume"
MODELS_DIR = "/root"
HF_DATASETS_CACHE = str(Path(VOLUME_DIR) / "hf_datasets_cache")
MODEL_CACHE = Path(VOLUME_DIR, "models")
assets_path = Path(__file__).parent /  "assets"
local_examples_path = Path(__file__).parent /  "videollava" / "serve" / "examples"
EXAMPLES_PATH = "/assets/examples"
mounts = [
    Mount.from_local_dir("./ai_video_editor/updated_video_llava", remote_path=REPO_HOME),
    Mount.from_local_dir(assets_path, remote_path="/assets"),
    Mount.from_local_dir(local_examples_path, remote_path=EXAMPLES_PATH),
]
volume = Volume.persisted("video-llava-vol")
volumes = {VOLUME_DIR: volume}
stub = Stub("updated-video-llava", mounts=mounts, volumes=volumes, secrets=[Secret.from_dotenv()])


image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    #Image.debian_slim()
    .apt_install(
        "git",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "clang",
        "libopenmpi-dev",
        gpu="any",
    )

    .pip_install(
        #  "torch==2.1.2",
        #  "transformers==4.37.2",
        #  "bitsandbytes==0.42.0",
    "torch==2.0.1", "torchvision==0.15.2",
    "transformers==4.31.0", "tokenizers>=0.12.1,<0.14", "sentencepiece==0.1.99", "shortuuid",
    "accelerate==0.21.0", "peft==0.4.0", "bitsandbytes==0.41.0",
    "pydantic<2,>=1", "markdown2[all]", "numpy", "scikit-learn==1.2.2",
    "requests", "httpx==0.24.0", "uvicorn", "fastapi",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
    "tensorboardX==2.6.2.2", "gradio==3.37.0", "gradio_client==0.7.0",
    "deepspeed==0.9.5", "ninja", "wandb",
        "wheel",
        gpu="any",
    )
    .run_commands(
        "python -m bitsandbytes",
        gpu="any"
    )
    #  .pip_install(
        #  "torchvision>=0.15.2",
        #  #"tokenizers>=0.12.1,<0.14",
        #  "sentencepiece==0.1.99",
        #  "shortuuid",
        #  "accelerate==0.21.0",
        #  "peft==0.4.0",
        #  "pydantic<2,>=1",
        #  "markdown2[all]",
        #  "numpy",
        #  "scikit-learn==1.2.2",
        #  "gradio==3.37.0",
        #  "gradio_client==0.7.0",
        #  "requests",
        #  "httpx==0.24.0",
        #  "uvicorn",
        #  "fastapi",
        #  #"einops==0.6.1",
        #  "einops-exts==0.0.4",
        #  "timm==0.6.13",
        #  "deepspeed==0.9.5",
        #  "ninja",
        #  "wandb",
        #  "tensorboardX==2.6.2.2",

        #  "tenacity",
        #  "torch==2.1.2",
        #  "wheel",
        #  gpu="any",
    #  )
    .run_commands("pip install flash-attn --no-build-isolation", gpu="any")
    .env({"PYTHONPATH": REPO_HOME, "HF_DATASETS_CACHE": HF_DATASETS_CACHE})
    .pip_install(
        "decord",
        "opencv-python",
        "git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d",
        gpu="any",
    )
    #.run_function(load_pretrained_from_cache, gpu="any")
)

def function_dec(**extras):
    return stub.function(
        image=image,
        timeout=80000,
        checkpointing_enabled=True,  # Enable memory checkpointing for faster cold starts.
        _allow_background_volume_commits=True,
        **extras,
    )

def cls_dec(**extras):
    return stub.cls(
        image=image,
        timeout=80000,
        checkpointing_enabled=True,  # Enable memory checkpointing for faster cold starts.
        **extras,
    )

def load_pretrained_from_cache(load_4bit=True, load_8bit=False):
    print("Loading pretrained model")
    from videollava.utils import disable_torch_init
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from videollava.model import LlavaLlamaForCausalLM
    import torch
    disable_torch_init()
    print("imported")

    kwargs = {
        "device_map": "auto",
        "cache_dir": HF_DATASETS_CACHE,
    }
    video_llava_path = Path(MODELS_DIR) / 'Video-LLaVA-7B'

    vlp_exists = video_llava_path.exists()
    if not vlp_exists:
        video_llava_path.mkdir(exist_ok=True, parents=True)

    save = False
    if not video_llava_path.exists() or len(list(video_llava_path.iterdir())) == 0:
        save = True
        print("Downloading model")
        video_llava_path = 'LanguageBind/Video-LLaVA-7B'

    tokenizer_path = Path(MODELS_DIR) / 'tokenizer'
    if not tokenizer_path.exists() or len(list(tokenizer_path.iterdir())) == 0:
        print("Downloading tokenizer")
        tokenizer_path = 'LanguageBind/Video-LLaVA-7B'

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, cache_dir=kwargs["cache_dir"])
    model = LlavaLlamaForCausalLM.from_pretrained(video_llava_path, low_cpu_mem_usage=True, **kwargs)
    model.generation_config.do_sample = True

    if save:
        # save to on-disk paths
        video_llava_path = Path(MODELS_DIR) / 'Video-LLaVA-7B'
        tokenizer_path = Path(MODELS_DIR) / 'tokenizer'
        tokenizer.save_pretrained(str(tokenizer_path))
        model.save_pretrained(str(video_llava_path))
    return model, tokenizer
