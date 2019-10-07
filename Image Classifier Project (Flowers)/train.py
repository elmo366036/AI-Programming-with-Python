import argparse
import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import sys
import buildmodel


# CONSTANTS
NUM_CATEGORIES = 102  # There are 102 flower categories.
ARCHITECTURES = ['densenet','resnet','alexnet','vgg']
CATEGORIES_FILENAME = 'cat_to_name.json'

def get_input_args():
    
    # Create argparser
    parser = argparse.ArgumentParser(description='train.py')

    # Set parser defaults
    parser.set_defaults(gpu=False)
    
    # Required arguments
    parser.add_argument('data_dir', type=str, action='store', help='path to folder of flower images')

    # Optional arguments
    parser.add_argument('--save_dir', dest='save_dir', type=str, action='store', help='path to save checkpoints')   
    parser.add_argument('--arch', dest='arch', action='store', choices=ARCHITECTURES, default = 'densenet', help='the CNN model architecture' )    
    parser.add_argument('--learning_rate', type=float, default = 0.001, help='learning rate for model training' )
    parser.add_argument('--dropout_probability', type=float, default = 0.2, help='dropout probability for model training' )
    parser.add_argument('--hidden_units', type=int, default = 512, help='number of hidden units for model training' )
    parser.add_argument('--epochs', type=int, default = 5, help='number of epochs for model training' )
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='use a gpu for numerical calculations')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    
    return parser.parse_args()
 
    
def validate_args(args):
    # Validate (some) args
    if not os.path.isdir(args.data_dir):
        raise Exception('\'' + args.data_dir + '\'' + ' is not a valid directory')
    if (args.save_dir is not None) and (not os.path.isdir(args.save_dir)):
        raise Exception('\'' + args.save_dir + '\'' + ' is not a valid directory')    
    if args.gpu and not torch.cuda.is_available():
        raise Exception('--gpu option not available')
    if args.arch not in ARCHITECTURES:
        raise Exception('--arch ' + '\'' + args.arch + '\'' + ' is not a valid architecture. Choose from ' + ARCHITECTURES)
    if not os.path.isfile(args.category_names):
        raise Exception('--category_names ' + '\'' + args.category_names + '\'' + ' is not a valid file') 
   

def create_dataloaders(data_dir):
    # Directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
        
    # Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder. Define as global variables
    global train_data
    global valid_data
    global test_data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define the dataloaders as global variables
    global trainloader
    global validloader
    global testloader
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
  
def get_num_outputs(category_names):
    # Determine the output size of the classifier from the json list of category names
    # or apply a default
    if (category_names is None) and (os.path.isfile(CATEGORIES_FILENAME)):
            with open(CATEGORIES_FILENAME, 'r') as f:
                cat_to_name = json.load(f)
    elif (category_names is None) and not (os.path.isfile(CATEGORIES_FILENAME)):
            output_size = NUM_CATEGORIES
    elif os.path.isfile(category_names):
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            
    output_size = max(map(int, cat_to_name.keys()))
    return output_size
    
# Functions to create an optimizer, create a criterion, and train a model
def create_optimizer(model, arch, learning_rate):
    models_with_classifier = ['densenet', 'alexnet', 'vgg']
    models_with_fc = ['resnet']
    
    if arch in models_with_classifier:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch in models_with_fc:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        raise EXCEPTION('train.py, create_optimizer   Cannot create optimizer')
    
    return optimizer    


def create_criterion():
    criterion = nn.NLLLoss()
    return criterion


def train_model(model, optimizer, criterion, epochs, gpu):    

    # Use GPU if available or if selected
    if gpu:
        device = 'cuda'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model tensors to default device
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 40

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
    
            # Move images and label tensors to default device
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f} ".format(accuracy/len(testloader)),
                      "Device: {}".format(device))          
            
                running_loss = 0
            
                # Make sure training is back on
                model.train()
            
    print('Done Training')

# Implement a function for the validation pass
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        # Move images and label tensors to default device
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # Calculate accuracy
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
    

# Save the checkpoint 
def save_model(model, optimizer, criterion, args, output_size):

    model.class_to_idx = train_data.class_to_idx
    model.to('cpu')
    
    if args.save_dir is None:
        path = args.arch + '_checkpoint.pth'
    else:
        path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
    
    checkpoint = {'architecture': args.arch,
                  'classifier_output_size': output_size,
                  'hidden_layer_size': args.hidden_units,
                  'learning_rate': args.learning_rate,
                  'dropout_probability': args.dropout_probability,
                  'epochs' : args.epochs,
                  'model_state': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'criterion_state': criterion.state_dict(),
                  'class_to_idx' : model.class_to_idx
                  }

    torch.save(checkpoint, path)


def main():
    args = get_input_args()
    validate_args(args)
    print(args)
    create_dataloaders(args.data_dir)   
    output_size = get_num_outputs(args.category_names)
    model = buildmodel.build_model(args.arch, output_size, args.hidden_units, args.dropout_probability)
    optimizer = create_optimizer(model, args.arch, args.learning_rate)
    criterion = create_criterion()
    train_model(model, optimizer, criterion, args.epochs, args.gpu)
    save_model(model, optimizer, criterion, args, output_size)


if __name__ == "__main__":
    main()
