CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
X_TOKEN_INDEX = {'IMAGE': -200, 'VIDEO': -201, 'AUDIO': -202, 'THERMAL': -203, 'DEPTH': -204}
X_INDEX_TOKEN = {v: k for k, v in X_TOKEN_INDEX.items()}
# IMAGE_TOKEN_INDEX = -200
DEFAULT_X_TOKEN = {'IMAGE': "<image>", 'VIDEO': "<video>", 'AUDIO': "<audio>", 'THERMAL': "<thermal>", 'DEPTH': "<depth>"}
# DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_X_PATCH_TOKEN = {'IMAGE': "<im_patch>", 'VIDEO': "<vi_patch>", 'AUDIO': "<au_patch>", 'THERMAL': "<th_patch>", 'DEPTH': "<de_patch>"}
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_X_START_TOKEN = {'IMAGE': "<im_start>", 'VIDEO': "<vi_start>", 'AUDIO': "<au_start>", 'THERMAL': "<th_start>", 'DEPTH': "<de_start>"}
# DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_X_END_TOKEN = {'IMAGE': "<im_end>", 'VIDEO': "<vi_end>", 'AUDIO': "<au_end>", 'THERMAL': "<th_end>", 'DEPTH': "<de_end>"}
# DEFAULT_IM_END_TOKEN = "<im_end>"
