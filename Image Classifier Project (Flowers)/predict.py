import numpy as np
import argparse
import os
import torch
import buildmodel
from PIL import Image
import json
import sys

# CONSTANTS
IMAGE_SHORTEST_SIDE = 256
IMAGE_CROP = 224
CATEGORIES_FILENAME = 'cat_to_name.json'

def get_input_args():
    
    # Create argparser
    parser = argparse.ArgumentParser(description='predict.py')

    # Set parser defaults
    parser.set_defaults(gpu=False)
    
    # Required arguments
    parser.add_argument('image_name', type=str, action='store', help='image and path to predict')
    parser.add_argument('checkpoint', type=str, action='store', help='filename of checkpoint')

    # Optional arguments
    parser.add_argument('--top_k', type=int, default=3, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='use a gpu for numerical calculations' )
    
    return parser.parse_args()
 
    
def validate_args(args):
    # Validate (some) args
    if not os.path.isfile(args.image_name):
        raise Exception('\'' + args.image_name + '\'' + ' is not a valid file')
    if not os.path.isfile(args.checkpoint):
        raise Exception('\'' + args.checkpoint + '\'' + ' is not a valid file')
    if not os.path.isfile(args.category_names):
        raise Exception('--category names ' + '\'' + args.category_names + '\'' + ' is not a valid file')    
    if args.gpu and not torch.cuda.is_available():
        raise Exception('--gpu option not available')


def load_checkpoint(filepath):
    # Load checkpoint and rebuild the model
    
    checkpoint = torch.load(filepath)
    
    architecture = checkpoint['architecture']
    classifier_output_size = checkpoint['classifier_output_size']
    hidden_layer_size = checkpoint['hidden_layer_size']
    learning_rate = checkpoint['learning_rate']
    dropout_probability = checkpoint['dropout_probability']
    epochs = checkpoint['epochs']
    model_state = checkpoint['model_state']
    optimizer_state = checkpoint['optimizer_state']
    criterion_state = checkpoint['criterion_state']
    class_to_idx = checkpoint['class_to_idx']
    
    model = buildmodel.build_model(architecture, classifier_output_size, hidden_layer_size, dropout_probability, class_to_idx)
    model.load_state_dict(model_state)
    print("Loaded: {} architecture={} # Hidden Units={} # Epochs={} Learning Rate={} Dropout= {}".format(filepath, 
                                                                                                          architecture, 
                                                                                                          hidden_layer_size, 
                                                                                                          epochs,
                                                                                                          learning_rate,
                                                                                                          dropout_probability))
    
    return model, class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    # Rescale so that smallest dimension is SHORTEST_SIDE. Maintain aspect ratio
    width, height = im.size
    scaling_factor = IMAGE_SHORTEST_SIDE / min(width, height)
    width_scaled, height_scaled = width*scaling_factor, height*scaling_factor
    im_scaled = im.resize((int(width_scaled), int(height_scaled)))
    
    # Crop
    left = int((width_scaled - IMAGE_CROP) / 2)
    right = int(left + IMAGE_CROP)
    top = int((height_scaled - IMAGE_CROP) / 2)
    bottom = int(top + IMAGE_CROP)
    im_scaled_cropped = im_scaled.crop((left, top, right, bottom))
    
    # Convert color channels from 0-255 to 0-1, normalize, transpose
    np_image = np.array(im_scaled_cropped) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = ((np_image - mean) / std).transpose((2,0,1))
    
    processed_image = normalized_image
    
    return processed_image  
 
                                     
def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    model.eval()    
    
    # Use GPU if available or if selected
    if gpu:
        device = 'cuda'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model tensors to default device
    model.to(device)
    
    # Process image and convert to tensor
    processed_image = process_image(image_path)
    processed_image_tensor = torch.from_numpy(processed_image)
    
    # Need to make the processed_image_tensor 1d and convert to FloatTensor
    processed_image_tensor = processed_image_tensor.unsqueeze_(0)
    processed_image_tensor = processed_image_tensor.type(torch.FloatTensor)
    
    # Move images tensor to default device
    processed_image_tensor = processed_image_tensor.to(device)
    
    # Get top k probabilities
    with torch.no_grad():
        output = model.forward(processed_image_tensor)
    ps_topk = torch.exp(output).topk(topk)   
    
    # Invert class_to_idx
    class_to_idx_inverted = dict(map(reversed, model.class_to_idx.items()))
    
    # Extract probabilities and classes from tensor, map classes to class_to_idx_inverted
    probs = (np.asarray(ps_topk[0])[0])
    idx = (np.asarray(ps_topk[1])[0])
    classes = [class_to_idx_inverted[idx[i]] for i in range(len(idx))]

    return probs, classes    
                                     

def get_category_names(category_names):
    # Get the category names. This will come either from a default file 
    # or from the command line argument if the file is available. 
    
    cat_file_available = False
    cat_to_name = ''
    if (category_names is None) and (os.path.isfile(CATEGORIES_FILENAME)):
            with open(CATEGORIES_FILENAME, 'r') as f:
                cat_to_name = json.load(f)
                cat_file_available = True                     
    elif (category_names is None) and not (os.path.isfile(CATEGORIES_FILENAME)):
            cat_to_name = None
            cat_file_available = False
    elif os.path.isfile(category_names):
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            cat_file_available = True
                                     
    return cat_to_name, cat_file_available


def display_results(probs, classes, cat_to_name, cat_file_available, top_k):
    if cat_file_available:
        flowers = [cat_to_name[classes[i]] for i in range(len(classes))]                             
    else:
        flowers = ['No names to match' for i in range(len(classes))]
        
    print()
    print('The top '+ str(top_k) + ' predictions are (in order of probability):')
    print()
    for i in range(top_k):
        print(str(i+1) + ') {} \t probability: {:.3f}'.format(flowers[i], probs[i]))
    print()
            
def main():
    args = get_input_args()
    validate_args(args)
    print(args)
    model, class_to_idx = load_checkpoint(args.checkpoint)                  
    probs, classes = predict(args.image_name, model, args.top_k, args.gpu)                             
    cat_to_name, cat_file_available = get_category_names(args.category_names)
    display_results(probs, classes, cat_to_name, cat_file_available, args.top_k)
                               

if __name__ == "__main__":
    main()