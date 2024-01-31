from modal import Volume, Image, Stub, Mount, Secret
from pathlib import Path
REPO_HOME = "/app"
VOLUME_DIR = "/volume"
MODELS_DIR = "/root"
HF_DATASETS_CACHE = str(Path(VOLUME_DIR) / "hf_datasets_cache")
mounts = [Mount.from_local_dir("./ai_video_editor/updated_video_llava", remote_path=REPO_HOME)]
volume = Volume.persisted("video-llava-vol")
volumes = {VOLUME_DIR: volume}
stub = Stub("updated-video-llava", mounts=mounts, volumes=volumes, secrets=[Secret.from_dotenv()])

image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
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
        "torch==2.1.2",
        "transformers==4.37.2",
        "bitsandbytes==0.42.0",
        gpu="any",
    )
    .run_commands(
        "python -m bitsandbytes",
        gpu="A10G"
    )
    .pip_install(
        "torchvision>=0.15.2",
        #"tokenizers>=0.12.1,<0.14",
        "sentencepiece==0.1.99",
        "shortuuid",
        "accelerate==0.21.0",
        "peft==0.4.0",
        "pydantic<2,>=1",
        "markdown2[all]",
        "numpy",
        "scikit-learn==1.2.2",
        "gradio==3.37.0",
        "gradio_client==0.7.0",
        "requests",
        "httpx==0.24.0",
        "uvicorn",
        "fastapi",
        #"einops==0.6.1",
        "einops-exts==0.0.4",
        "timm==0.6.13",
        "deepspeed==0.9.5",
        "ninja",
        "wandb",
        "tensorboardX==2.6.2.2",

        "tenacity",
        "torch==2.1.2",
        "wheel",
        gpu="any",
    )
    .run_commands("pip install flash-attn --no-build-isolation", gpu="any")
    .env({"PYTHONPATH": REPO_HOME, "HF_DATASETS_CACHE": HF_DATASETS_CACHE})
    .pip_install(
        "decord",
        "opencv-python",
        # TODO try removing or upgrading this to a version
        "git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d",
        gpu="any",
    )
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
