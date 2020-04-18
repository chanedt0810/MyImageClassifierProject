import numpy as np
import torch
from torchvision import models
from PIL import Image
import json
import argparse

# Creating argparser and adding the values
args_parser = argparse.ArgumentParser(description='Training of Network')
args_parser.add_argument('image_path', type=str, action='store', help='required - path to image: "path/filename" format')
args_parser.add_argument('checkpoint_file', type=str, action='store', help='required - checkpoint filename')
args_parser.add_argument('--top_k', type=int, action='store', default=1, help='optional - top number of probabilities')
args_parser.add_argument('--category_names', type=str, action='store', default='', help='optional - name of file containing category names')
args_parser.add_argument('--gpu', type=bool, action='store', default=False, help='optional set True if gpu is to be used')

# Getting the input arguments arguments
in_args = args_parser.parse_args()

image_path = in_args.image_path
checkpoint_file = in_args.checkpoint_file
top_k = in_args.top_k
category_names = in_args.category_names
gpu = in_args.gpu

device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

if category_names != '':
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

# Method to load the checkpoint file        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['idx_to_class']
    model.learning_rate = checkpoint['learning_rate']
    
    return model

# method to process the image file
def process_image(image):
    image = Image.open(image)
    original_image_width, original_image_height = image.size
    ar = original_image_width/original_image_height
    max_size = 256
    crop_size = 224
    divisor = 2
    
    if ar > 1:
        image = image.resize((round(ar*max_size), max_size))
    else:
        image = image.resize((max_size, round(ar*max_size)))
    
    new_image_width, new_image_height = image.size    
    
    left = (new_image_width - crop_size)/divisor    
    right = (new_image_width + crop_size)/divisor 
    top = (new_image_height - crop_size)/divisor
    bottom = (new_image_height + crop_size)/divisor
    
    image = image.crop((left, top, right, bottom))
    image = np.array(image)/255
    
    means = np.array([0.485, 0.456, 0.406])
    std_deviations = np.array([0.229, 0.224, 0.225])
    
    image = (image - means)/std_deviations
    image = np.transpose(image, (2,0,1))
    
    return image

# method to calculate the probabilities of the associated classes
def predict(image_path, model, device, topk):
    model.to(device)
    
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)
    image = image.unsqueeze(0)    
    with torch.no_grad():
        image_output = model.forward(image.to(device))
    probability = torch.exp(image_output)
    probs = torch.topk(probability, topk)[0]     
    probs = probs.cpu().detach().numpy().tolist()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[int(i)] for i in np.array(torch.topk(probability, topk)[1][0])]
    
    return probs[0], classes

model = load_checkpoint(checkpoint_file)
probs, classes = predict(image_path, model, device, top_k)


results_dict = dict()
for index in range(0, len(classes), 1):
    if category_names == '':
        class_item = classes[index]
    else:
        class_item = cat_to_name[classes[index]]
        
    results_dict[class_item] = probs[index]
    
print(results_dict)