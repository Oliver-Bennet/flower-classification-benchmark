import timm

def create_swin(num_classes=5, model_name='swin_tiny_patch4_window7_224', pretrained=False):
    """
    Swin Transformer Tiny - rất mạnh trên image classification.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model