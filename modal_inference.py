from modal import (
    Cls,
    method,
    enter,
    web_endpoint,
)
from pathlib import Path
from .stub import stub, cls_dec, function_dec, MODELS_DIR, volume, HF_DATASETS_CACHE, REPO_HOME, load_pretrained_from_cache


LOCAL_VIDEOS_PATH = Path(REPO_HOME) / "downloaded_videos"
DEFAULT_PROMPT = "describe what is going on in this video"


#  def load_pretrained_from_cache(load_4bit=True, load_8bit=False):
    #  print("Loading pretrained model")
    #  from videollava.utils import disable_torch_init
    #  from transformers import AutoTokenizer, BitsAndBytesConfig
    #  from videollava.model import LlavaLlamaForCausalLM
    #  import torch
    #  disable_torch_init()
    #  print("imported")

    #  kwargs = {
        #  "device_map": "auto",
        #  "cache_dir": HF_DATASETS_CACHE,
    #  }
    #  video_llava_path = Path(MODELS_DIR) / 'Video-LLaVA-7B'

    #  vlp_exists = video_llava_path.exists()
    #  if not vlp_exists:
        #  video_llava_path.mkdir(exist_ok=True, parents=True)

    #  save = False
    #  if not video_llava_path.exists() or len(list(video_llava_path.iterdir())) == 0:
        #  save = True
        #  print("Downloading model")
        #  video_llava_path = 'LanguageBind/Video-LLaVA-7B'

    #  tokenizer_path = Path(MODELS_DIR) / 'tokenizer'
    #  if not tokenizer_path.exists() or len(list(tokenizer_path.iterdir())) == 0:
        #  print("Downloading tokenizer")
        #  tokenizer_path = 'LanguageBind/Video-LLaVA-7B'

    #  if load_8bit:
        #  kwargs['load_in_8bit'] = True
    #  elif load_4bit:
        #  kwargs['load_in_4bit'] = True
        #  kwargs['quantization_config'] = BitsAndBytesConfig(
            #  load_in_4bit=True,
            #  bnb_4bit_compute_dtype=torch.float16,
            #  bnb_4bit_use_double_quant=True,
            #  bnb_4bit_quant_type='nf4'
        #  )
    #  else:
        #  kwargs['torch_dtype'] = torch.float16

    #  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, cache_dir=kwargs["cache_dir"])
    #  model = LlavaLlamaForCausalLM.from_pretrained(video_llava_path, low_cpu_mem_usage=True, **kwargs)
    #  model.generation_config.do_sample = True

    #  if save:
        #  # save to on-disk paths
        #  video_llava_path = Path(MODELS_DIR) / 'Video-LLaVA-7B'
        #  tokenizer_path = Path(MODELS_DIR) / 'tokenizer'
        #  tokenizer.save_pretrained(str(tokenizer_path))
        #  model.save_pretrained(str(video_llava_path))
    #  return model, tokenizer

def prepare_processor(model, device="cuda"):
    import torch
    processor = {'image': None, 'video': None}
    if model.config.mm_image_tower is not None:
        image_tower = model.get_image_tower()
        if not image_tower.is_loaded:
            image_tower.load_model()
        image_tower.to(device=device, dtype=torch.float16)
        image_processor = image_tower.image_processor
        processor['image'] = image_processor
    if model.config.mm_video_tower is not None:
        video_tower = model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=device, dtype=torch.float16)
        video_processor = video_tower.video_processor
        processor['video'] = video_processor
    return processor

def prepare_special_tokens(model, tokenizer):
    from videollava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
        DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

@cls_dec(container_idle_timeout=30, gpu='L4')
class VideoLlava:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
    # TODO when they fix
    #@build()
    @enter()
    def load_model(self):
        self.model, self.tokenizer = load_pretrained_from_cache(load_4bit=False, load_8bit=True)
        print("got model")
        self.processor = prepare_processor(self.model, device=self.device)
        self.video_processor = self.processor['video']

    def prepare_conv(self):
        from videollava.conversation import conv_templates
        self.conv = conv_templates["llava_v1"].copy()
        self.roles = self.conv.roles

    @method()
    def inference(self, video_path: str, inp: str = DEFAULT_PROMPT):
        import torch
        from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from videollava.conversation import SeparatorStyle
        import requests
        from io import BytesIO
        print('preparing conv')

        self.prepare_conv()
        if video_path.startswith("http"):
            print("Downloading video")
            video_bytes = requests.get(video_path).content
            local_video_path = LOCAL_VIDEOS_PATH / video_path.split("/")[-1]
            if not LOCAL_VIDEOS_PATH.exists():
                LOCAL_VIDEOS_PATH.mkdir(exist_ok=True, parents=True)
            with open(local_video_path, "wb") as f:
                f.write(video_bytes)
            video_path = BytesIO(video_bytes)
            print(f"Downloaded video and saved to {local_video_path}")
        elif not Path(video_path).exists():
            volume.reload()
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video {video_path} not found")
        print('processing video')
        video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(self.model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(self.model.device, dtype=torch.float16)

        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        print(input_ids.shape, tensor.shape)

        import torch
        import time
        begin = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        end = time.time()
        print(f"Generate time taken: {end-begin}")

        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(output)
        return output

@function_dec(gpu='L4')
@web_endpoint()
def inference(video_path: str, inp: str):
    video_llava = VideoLlava()
    if not hasattr(video_llava, 'model') or video_llava.model is None:
        print('loading model')
        video_llava.load_model()
    print('model loaded')
    output = VideoLlava().inference(video_path, inp)
    print(output)
    return output

@stub.local_entrypoint()
def main():
    prompt = "describe what is going on in this video"
    video_path = '/volume/pika_water_city.mp4'
    input_ids, tensor, stopping_criteria = VideoLlava().get_inputs.remote(video_path, prompt)
    output = VideoLlava().inference.remote(input_ids, tensor, stopping_criteria)
    print(output)

if __name__ == "__main__":
    video_llava = Cls.lookup("updated-video-llava-ephemeral", 'VideoLlava')()
    video_path = '/volume/pika_water_city.mp4'
    #print(video_llava.inference.remote(video_path=video_path, inp="describe what is going on in this video"))

    prompt = "describe what is going on in this video"
    video_path = '/volume/pika_water_city.mp4'
    output = video_llava.inference.remote(video_path, prompt)
    print(output)

