import timm

def create_maxvit(num_classes=5, model_name='maxvit_tiny_tf_224', pretrained=False):
    """
    MaxViT Tiny - hybrid mạnh mẽ (CNN + Transformer + Grid Attention).
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model