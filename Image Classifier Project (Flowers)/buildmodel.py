import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_model(arch):
    # Load a pre-trained model such as DenseNet
    
    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
    elif arch == 'resnet':
        model = models.resnet18(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        raise Exception('get_model: undefined architecture')
    
    return model

def build_model(arch, classifier_output_size, hidden_layer_size, dropout_prob, class_to_idx = []):
    # Update classifier input size based on model type
    
    # Get the model
    model = get_model(arch)
 
    # Parameters
    if arch == 'densenet':
        classifier_input_size = model.classifier.in_features
    elif arch == 'resnet':
        classifier_input_size = model.fc.in_features 
    elif arch == 'alexnet':
        classifier_input_size = model.classifier[1].in_features
    elif arch == 'vgg':
        classifier_input_size = model.classifier[0].in_features
    
    # Check to see if model was loaded or is new
    loaded = False
    if class_to_idx != []:
        loaded = True

    # Freeze parameters in that pre-trained model so we don't backprop through them
    # i.e., we do not attempt to optimize it
    for param in model.parameters():
        param.requires_grad = False   

    # Create a new classifier    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input_size, hidden_layer_size)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(dropout_prob)),
                          ('fc2', nn.Linear(hidden_layer_size, classifier_output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Replace the classifier of the pre-trained model with the updated one    
    models_with_classifier = ['densenet', 'alexnet', 'vgg']
    models_with_fc = ['resnet']
    
    if arch in models_with_classifier:
        model.classifier = classifier
    elif arch in models_with_fc:
        model.fc = classifier
    else:
        raise EXCEPTION('buildmodel.py   Cannot replace model classifier')
    
    # If model was loaded, replace class_to_idx with the one that was loaded
    if loaded:
        model.class_to_idx = class_to_idx
        
    return model 