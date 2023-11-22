import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
from .mae_encoder import MAEVisionTower
from transformers import CLIPModel

CACHE_DIR = os.getenv('VIDEO_LLAVA_CACHE_DIR','./cache_dir')

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir=CACHE_DIR, **kwargs)
    if 'mae' in image_tower:
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        return MAEVisionTower(image_tower, args=image_tower_cfg, cache_dir=CACHE_DIR, **kwargs)
    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir=CACHE_DIR, **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')



# import os
# from .clip_encoder import CLIPVisionTower
# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
# from transformers import CLIPModel

# def build_image_tower(image_tower_cfg, **kwargs):
#     image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
#     is_absolute_path_exists = os.path.exists(image_tower)
#     if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
#         return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
#     if image_tower.endswith('LanguageBind_Image'):
#         return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir=CACHE_DIR', **kwargs)
#     raise ValueError(f'Unknown image tower: {image_tower}')

# def build_video_tower(video_tower_cfg, **kwargs):
#     video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
#     if video_tower.endswith('LanguageBind_Video'):
#         return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir=CACHE_DIR, **kwargs)
#     raise ValueError(f'Unknown video tower: {video_tower}')