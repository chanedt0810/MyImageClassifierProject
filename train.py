import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

# Creating argparser and adding the values
args_parser = argparse.ArgumentParser(description='Training of Network')
args_parser.add_argument('data_dir', type=str, action='store', help='required path for flower images folder: "path/" format')
args_parser.add_argument('--save', type=str, action='store', default='', help='optional path for saving of the checkpoint.pth file:  "path/" format')
args_parser.add_argument('--arch', type=str, action='store', default='vgg16', help='optional pretrained architecture with vgg16 set as the default')
args_parser.add_argument('--learning_rate', type=float, action='store', default=0.001, help='optional learning rate with 0.001 set as the default')
args_parser.add_argument('--hidden_units', type=int, action='store', default=512, help='optional hidden layer units with 512 set as the default')
args_parser.add_argument('--epochs', type=int, action='store', default=10, help='optional training epochs with 10 set as the default')
args_parser.add_argument('--gpu', type=bool, action='store', default=False, help='optional set True if gpu is to be used')

# Getting the input arguments arguments
in_args = args_parser.parse_args()

data_dir = in_args.data_dir
checkpoint_path = in_args.save
arch = in_args.arch
learn_rate = in_args.learning_rate
hidden_units = in_args.hidden_units
epochs = in_args.epochs
gpu = in_args.gpu

train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'

input_size = 25088
output_size = 102

device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

# Method to get the dataloaders and image datasets for a given directory with batchsize
def get_dataloaders(transforms, directory, batch_size=32, shuffle=False, train=False):
    if train:
        data_transforms = transforms.Compose([transforms.RandomRotation(20),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    else:
        data_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(directory, transform=data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=shuffle)
    
    return dataloaders, image_datasets

# Method to set the model architecture
def set_architecture(arch):
    if arch == 'vgg11':
        return models.vgg11(pretrained=True), 25088;
    elif arch == 'vgg13':
        return models.vgg13(pretrained=True), 25088;
    elif arch == 'vgg16':
        return models.vgg16(pretrained=True), 25088;
    elif arch == 'vgg19':
        return models.vgg19(pretrained=True), 25088;
    elif arch == 'alexnet':
        return models.alexnet(pretrained=True), 9216;
    elif arch == 'densenet121':
        return models.densenet121(pretrained=True), 1024;
    elif arch == 'densenet169':
        return models.densenet169(pretrained=True), 1024;
    elif arch == 'densenet161':
        return models.densenet161(pretrained=True), 1024;
    elif arch == 'densenet201':
        return models.densenet201(pretrained=True), 1024;

# Method to build the model    
def build_model(model, device, input_size, learning_rate, hidden_units):
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_units, output_size),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    return model, criterion, optimizer 

# Method to train the model
def train(model, optimizer, criterion, epochs, train_dataloaders, valid_dataloaders, device):
    steps = 0
    running_loss = 0
    print_every = 20
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for train_inputs, train_labels in train_dataloaders:
            steps += 1
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

            optimizer.zero_grad()        
            train_output = model.forward(train_inputs)
            train_loss = criterion(train_output, train_labels)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for valid_inputs, valid_labels in valid_dataloaders:
                        valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                        valid_output = model.forward(valid_inputs)
                        valid_loss += criterion(valid_output, valid_labels).item()

                        probabilities = torch.exp(valid_output)
                        equality = (valid_labels == probabilities.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()                    

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training loss: {train_loss/len(train_dataloaders):.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_dataloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_dataloaders):.3f}.. ")
                running_loss = 0
                model.train()

    time_end = time.time() - start
    print(f"\nTraining completed: {time_end//60:.0f}m {time_end%60:0f}s")

# Method to save trained model to checkpoint file
def save_checkpoint(model, optimizer, checkpoint_path, arch, input_size, output_size, learning_rate):
    checkpoint = {'architecture': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'idx_to_class': model.class_to_idx,
                  'learning_rate': learning_rate}

    torch.save(checkpoint, checkpoint_path + 'checkpoint.pth')

train_dataloaders, train_data = get_dataloaders(transforms, train_dir, 32, shuffle=True, train=True)
valid_dataloaders, valid_data = get_dataloaders(transforms, valid_dir, 32)

model, input_size = set_architecture(arch)
model, criterion, optimizer = build_model(model, device, input_size, learn_rate, hidden_units)
train(model, optimizer, criterion, epochs, train_dataloaders, valid_dataloaders, device)

model.class_to_idx = train_data.class_to_idx

save_checkpoint(model, optimizer, checkpoint_path, arch, input_size, output_size, learn_rate)