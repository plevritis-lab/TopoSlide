'''
    Adapted from https://huggingface.co/MahmoodLab/TITAN/blob/main/configuration_titan.py
'''

import copy
from typing import Any

from transformers import PretrainedConfig

class VisionConfig(PretrainedConfig):
    model_type = "vision_transformer"
    
    def __init__(
        self,
        grid_size: int =14,
        global_pool: str ='token',
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        mlp_patch_embed_dim: int =768,
        pos_encode_type: str ='alibi',
        #### CoCa params ####
        attentional_pool: str = None,
        attn_pooler_queries: int = 128,
        attn_pooler_heads: int = 8,
        #### shahira added ####
        return_all_tokens: bool = False, 
        return_all_tokens_before_pooling: bool = False, 
        return_all_tokens_coord: bool = False, 
        **kwargs: Any,
    ):
        self.grid_size = grid_size
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.mlp_patch_embed_dim = mlp_patch_embed_dim
        self.pos_encode_type = pos_encode_type
        self.attentional_pool = attentional_pool
        self.attn_pooler_queries = attn_pooler_queries
        self.attn_pooler_heads = attn_pooler_heads
        self.return_all_tokens = return_all_tokens
        self.return_all_tokens_before_pooling = return_all_tokens_before_pooling
        self.return_all_tokens_coord = return_all_tokens_before_pooling

        super().__init__(**kwargs)



class TopoSlideConfig(PretrainedConfig):
    model_type = "toposlide"
    is_composition = True

    def __init__(
        self,
        vision_config: VisionConfig = VisionConfig(),
        **kwargs: Any,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        ### for CoCa ###
        self.vision_config.attentional_pool = "parallel"

        super().__init__(**kwargs)
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        for k, v in output.items():
            if isinstance(v, PretrainedConfig):
                output[k] = v.to_dict()
        output["model_type"] = self.__class__.model_type
        return output