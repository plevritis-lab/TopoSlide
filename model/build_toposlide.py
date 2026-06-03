'''
    Adapted from https://huggingface.co/MahmoodLab/TITAN/blob/main/modeling_titan.py
'''
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .vision_transformer import build_vision_tower
from .configuration import TopoSlideConfig

class TopoSlide(PreTrainedModel):
    config_class = TopoSlideConfig

    def __init__(self, config: TopoSlideConfig, *model_args, **model_kwargs):
        super().__init__(config)
        # print('config.vision_config\n', config.vision_config)
        self.vision_encoder = build_vision_tower(config.vision_config)

    def encode_slide_from_patch_features(self, patch_features: torch.Tensor, patch_coords: torch.Tensor, patch_size_lv0: int) -> torch.Tensor:
        '''
        encode whole-slide image using patch features
        Args:
            patch_features: torch.Tensor, shape (1, N, C)
            patch_coords: torch.Tensor, shape (1, N, 2)
            patch_size_lv0: int, patch size at level 0 (1024 if slide is 40x, 512 if slide is 20x)
        '''
        slide_embedding = self.vision_encoder(patch_features, patch_coords, patch_size_lv0, no_proj=True)
        return slide_embedding
    
    def forward_features_patch_embed(self, patch_features: torch.Tensor) -> torch.Tensor:
        '''
        encode whole-slide image using patch features
        Args:
            patch_features: torch.Tensor, shape (N, C)
        '''
        patch_embeddings = self.vision_encoder.forward_features_patch_embed(patch_features)
        return patch_embeddings

