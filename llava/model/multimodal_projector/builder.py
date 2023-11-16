from typing import Optional

import torch
import torch.nn as nn
import re

from transformers import PretrainedConfig, Blip2PreTrainedModel, Blip2Config, Blip2QFormerModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


# def build_vision_projector(config, delay_load=False, **kwargs):
#     projector_type = getattr(config, 'mm_projector_type', 'linear')
#
#     if projector_type == 'linear':
#         return nn.Linear(config.mm_hidden_size, config.hidden_size)
#
#     mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
#     if mlp_gelu_match:
#         mlp_depth = int(mlp_gelu_match.group(1))
#         modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
#         for _ in range(1, mlp_depth):
#             modules.append(nn.GELU())
#             modules.append(nn.Linear(config.hidden_size, config.hidden_size))
#         return nn.Sequential(*modules)
#
#     if projector_type == 'identity':
#         return IdentityMap()
#
#     raise ValueError(f'Unknown projector type: {projector_type}')


class Blip2Model(Blip2PreTrainedModel):
    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # self.proj = nn.Linear(config.mm_hidden_size, config.hidden_size)
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]
        self.proj = nn.Sequential(*modules)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # vision_outputs = self.vision_model(
        #     pixel_values=pixel_values,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # image_embeds = vision_outputs[0]
        # image_embeds = self.proj(pixel_values)
        image_embeds = pixel_values


        # print('pixel_values to proj', pixel_values.shape, image_embeds.shape)
        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        # print('qformer out', query_outputs.shape)
        query_outputs = self.proj(query_outputs)
        return query_outputs


def qformer_config_template(config, projector_type):
    pattern = r"qformer(\d+)_(\d+)"

    match = re.search(pattern, projector_type)
    num_hidden_layers = int(match.group(1))
    num_query_tokens = int(match.group(2))

    qformer_config = type('Blip2Config', (PretrainedConfig,), {
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "model_type": "blip-2",
        "num_query_tokens": num_query_tokens,
        "hidden_size": config.hidden_size,
        "mm_hidden_size": config.mm_hidden_size,
        "qformer_config": type('qformer_config', (PretrainedConfig,), {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": None,
            "cross_attention_frequency": 1,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "encoder_hidden_size": config.mm_hidden_size,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": config.mm_hidden_size,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "initializer_range": 0.02,
            "intermediate_size": config.mm_hidden_size * 4,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
            },
            "layer_norm_eps": 1e-12,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 512,
            "min_length": 0,
            "model_type": "blip_2_qformer",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 32,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": num_hidden_layers,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.27.0.dev0",
            "typical_p": 1.0,
            "use_bfloat16": False,
            "vocab_size": 30522
        })()
    })()
    return qformer_config

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type == 'identity':
        return IdentityMap()

    elif projector_type.startswith('qformer'):  # qformer2_64
        qformer_config = qformer_config_template(config, projector_type)
        return Blip2Model(qformer_config)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')