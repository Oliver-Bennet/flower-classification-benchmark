import timm

def create_convnext(num_classes=5, model_name='convnext_tiny', pretrained=False):
    """
    ConvNeXt Tiny - modern CNN, thường rất mạnh.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model