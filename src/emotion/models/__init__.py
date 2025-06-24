from .emotionnet import *
from .resnet import *
from .faceemotioncnn import *
from .convnet import *
from .cnn import *
from .vgg import *
from .resemotion import *
from .efficientnet import EfficientNet, VALID_MODELS
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

def getModel(model_name=None, modelName=None, num_classes=1000, silent=False):
    # 하위 호환성을 위해 두 매개변수 모두 지원
    if model_name is not None:
        modelName = model_name
    elif modelName is None:
        modelName = "efficientnet-b5"  # last.py와 동일한 기본값
        
    modelName = modelName.lower()
    if modelName == "vgg19":
        if not silent: print("Model - VGG19")
        return VGG("VGG19", num_classes=num_classes)
    elif modelName == "vgg22":
        if not silent: print("Model - VGG22")
        return VGG("VGG22", num_classes=num_classes)
    elif modelName == "vgg24":
        if not silent: print("Model - VGG24")
        return VGG("VGG24", num_classes=num_classes)
    elif modelName == "resnet18":
        if not silent: print("Model - ResNet18")
        return ResNet18(num_classes=num_classes)
    elif modelName == "emotionnet":
        if not silent: print("Model - EmotionNet")
        return EmotionNet(num_classes=num_classes)
    elif modelName == "resemotionnet":
        if not silent: print("Model - ResEmotionNet")
        return ResEmotionNet(num_classes=num_classes)
    elif modelName == "efficientnet-b4":
        if not silent: print("Model - EfficientNet-b4")
        model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
        return model
    elif modelName == "efficientnet-b5":
        if not silent: print("Model - EfficientNet-b5")
        model = EfficientNet.from_name('efficientnet-b5', num_classes=num_classes)
        return model

    if not silent: 
        print("Invalid model input:", modelName)
        print("Use instead: CNN")
        
    return CNN()