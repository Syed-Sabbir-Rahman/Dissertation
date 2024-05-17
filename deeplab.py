from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

def get_deeplab(outputchannels):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    
    # for i,param in enumerate(model.parameters()):
    #     param.requires_grad = False
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    # model.train()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.n_classes = outputchannels

    print("Total: {}, trainable: {}".format(pytorch_total_params,pytorch_total_trainable_params))
    return model