from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES
# from torchvision.models import efficientnet 
# from torchvision.models import vision_transformer 
def build_backbone(key, multi_scale=False):

    model_dict = {
        'resnet34': 512,
        'resnet18': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'tresnet': 2432,
        'swin_s': 768,
        'swin_b': 1024,
        'vit_s': 768,
        'vit_b_16': 768,
        'bninception': 1024,
        'tresnetM': 2048,
        'tresnetL': 2048,
        'efficientnet_v2_s': 1280,
        'efficientnet_b4': 1729,
        'efficientnet_v2_m': 1280,
    }
    #print(key)
    model = BACKBONE[key]()
    #efficientnet.model_urls["efficientnet_b3"] = "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth"
   
    # Initialize the model using the legacy API
    #model = efficientnet.efficientnet_v2_s(pretrained=True)
    #model = vision_transformer.vit_b_16(pretrained=True)
    #print(model)
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

