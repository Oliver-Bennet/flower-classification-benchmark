import timm
import torch.nn as nn

def create_vit(num_classes=5, model_name='vit_tiny_patch16_224', pretrained=False):
    """
    Tạo Vision Transformer từ timm.
    Dùng vit_tiny để train nhanh trên dataset nhỏ.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model