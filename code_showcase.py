######################################################################################################################
# This is a code a sample exploration of PyTorch, specifically a 
# Densely Connected Convolutional Network for noisy spectrogram classification problem
# 
# It should be noted that this code is a demonstration of technical skills and approach
# The code has not been optimized in terms of memory/processing efficiency nor
# optimized in terms of model parameters
#
# This proof of concept runs, but crashes during training on my local machine
# I am currently in the process of acquiring more computational resources to process this
#
# Without the added scaling and augmentation below, the unrefined model on un-preprocessed data returns an
# accuracy of:
# L1 Testing Accuracy - 0.6581
# H1 Testing Accuracy - 0.6577 
#
# More details on the context of the data and issue at hand can be consulted in this repo at code_writeup.md
# 
# Compiled by Jack Hayes on November 28, 2023
######################################################################################################################

# imports
import pandas as pd
from pathlib import Path
import numpy as np
import glob
import h5py
import os

import warnings
warnings.filterwarnings('ignore')

from IPython.display import HTML, display
display(HTML('<style>.font-family:verdana; word-spacing:1.5px;</style>'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image

# define local data paths 
# note that i use the train data as test data as well, as i do not have test labels since the data is from Kaggle where
# test labels are provided among submission
data_path = Path(r"C:\Users\JackE\data\g2net\data\train")
labels_path = Path(r"C:\Users\JackE\data\g2net\data\train_labels.csv")
# read in the labels
train_labels = pd.read_csv(labels_path)

# function calculate min and max values for a given SFTs data
# libraries such as scikit-learn struggle scaling data this small, thus a manual approach was taken
def calculate_min_max(data_path, labels):
    '''
    Extracts data from an hdf5 file given a path and makes each group/data its own key value pair
        data_path: filepath to data parent directory
        labels: training_labels data as a pd dataframe
    Returns: float, float: minimum and maximum values of the entire dataset
    '''
    min_val = float('inf')
    max_val = float('-inf')

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        with h5py.File(file_path, "r") as f:
            id_key = list(f.keys())[0]
            
            # skip files with label equal to -1
            # the creators of this Kaggle competition included 'easter eggs'
            # where some samples have labels of -1 and are fun pictures rather than the data of interest
            label = labels.loc[labels.id == id_key].target.item()
            if label == -1:
                continue

            # skip files with timestamps less than 4096
            # we want to ensure all spectrograms have the same shape
            # 4096 was settled on by manually looking at the timestamp lengths of the files
            # this number excludes outlier files which have sginifcantly less timestamp counts
            l1_timestamps = np.array(f[id_key]['L1']['timestamps_GPS'])
            h1_timestamps = np.array(f[id_key]['H1']['timestamps_GPS'])
            if len(l1_timestamps) < 4096 or len(h1_timestamps) < 4096:
                continue

            # extract the 'real part' from SFTs rather than the imaginary part
            l1_sfts = np.real(np.array(f[id_key]['L1']['SFTs']))
            h1_sfts = np.real(np.array(f[id_key]['H1']['SFTs'])) 

            file_min = np.min([np.min(l1_sfts), np.min(h1_sfts)])
            file_max = np.max([np.max(l1_sfts), np.max(h1_sfts)])

            min_val = min(min_val, file_min)
            max_val = max(max_val, file_max)

    return min_val, max_val

# get min and max values of entire dataset to scale the values later
min_dataset_val, max_dataset_val = calculate_min_max(data_path, train_labels)

# function to extract data from hdf5 format
def extract_data_from_hdf5(path, labels):
    '''
    Extracts data from an hdf5 file given a path and makes each group/data its own key value pair
        path: filepath to an individual hdf5 file in the dataset
        labels: training_labels data as a pd dataframe
    Returns: dictionary: The frequency, label, and respective L1 and H1 timestamps and SFTs
    '''
    data = {}
    
    with h5py.File(path, "r") as f:
        id_key = list(f.keys())[0]

        # retrieve the frequency data
        data['freq'] = np.array(f[id_key]['frequency_Hz'])

        # retrieve the Livingston detector data
        data['L1_SFTs'] = np.real(np.array(f[id_key]['L1']['SFTs']))
        data['L1_ts'] = np.array(f[id_key]['L1']['timestamps_GPS'])

        # retrieve the Hanford detector data
        data['H1_SFTs'] = np.real(np.array(f[id_key]['H1']['SFTs']))
        data['H1_ts'] = np.array(f[id_key]['H1']['timestamps_GPS'])
        
        # get label from training labels if in the training set
        data['label'] = labels.loc[labels.id == id_key].target.item()

    # return the dictionary 
    return data


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms_list, num_images_list, transform=None):
        '''
        Custom Dataset class for handling spectrogram data, labels, and number of images
            spectrograms_list: list of dictionaries, each containing spectrogram data and associated metadata
            num_images_list: list of integers representing the number of images for each spectrogram
            transform: data augmentation transform
        '''
         # store the list of spectrograms, number of images, and the data augmentation transformation
        self.spectrograms_list = spectrograms_list
        self.num_images_list = num_images_list
        self.transform = transform

    def __len__(self):
        '''
        Returns the number of spectrograms in the dataset
        '''
        return len(self.spectrograms_list)

    def __getitem__(self, idx):
        '''
        Returns a dictionary containing the spectrogram, label, and number of images for the spectrogram at the given index
            idx: index of the spectrogram to retrieve.
        '''
        # extract the spectrogram, label, and num_images for the given index
        spectrogram = self.spectrograms_list[idx]['spectrogram']
        label = self.spectrograms_list[idx]['label']
        num_images = self.num_images_list[idx]

        # convert the input spectrogram to a PIL image for spectrogram augmentation preprocessing
        img_pil = Image.fromarray(spectrogram)

        if self.transform:
             # apply data augmentation if the transform is provided
            augmented_img_pil = self.transform(img_pil)
             # convert the augmented PIL image back to a numpy array so we can make it a float32
             # so it can 'play nice' with pytorch
            augmented_spectrogram = np.array(augmented_img_pil).astype(np.float32)
        else:
            # else if no transformation defined
            # no augmentation needed for testing data
            augmented_spectrogram = spectrogram

        # add a batch dimension to the spectrogram and convert to a pytorch tensor
        augmented_spectrogram = torch.tensor(augmented_spectrogram, dtype=torch.float).unsqueeze(0)

        return {'spectrogram': augmented_spectrogram, 'label': label, 'num_images': num_images}

def apply_transform(transform, spectrogram):
    """
    Applies the given transformation to the input spectrogram.
        transform (torchvision.transforms.Transform): The transformation to apply.
        spectrogram (torch.Tensor): The input spectrogram.
    Returns: torch.Tensor: The augmented spectrogram.
    """
    # squeeze if the spectrogram has a batch dimension
    if spectrogram.ndim == 3 and spectrogram.shape[0] == 1:
        spectrogram = np.squeeze(spectrogram, axis=0)
    # convert the spectrogram to a PIL Image
    img_pil = transforms.ToPILImage()(spectrogram.astype(np.uint8))
    # apply the transformation
    augmented_img_pil = transform(img_pil)
    # convert the augmented PIL Image back to a numpy.ndarray
    augmented_spectrogram = np.array(augmented_img_pil).astype(np.float32)

    return augmented_spectrogram

def create_splits(data, image_size=4096, transform=None):
    '''
    Creates splits of spectrogram data based on the given image size
        data: dict containing the spectrogram data and metadata
        image_size: size of each split spectrogram
    Returns: list,list: list of L1 and H1 spectrograms respectively
    '''
    spectrograms_list_L1 = []
    spectrograms_list_H1 = []

    L1_ts = data['L1_ts']
    L1_SFTs = data['L1_SFTs']
    H1_ts = data['H1_ts']
    H1_SFTs = data['H1_SFTs']
    label = data['label']

    # define how many spectrograms will be subsetted from a single sample
    num_images_L1 = len(L1_ts) // image_size
    num_images_H1 = len(H1_ts) // image_size

    # skip samples with a label of -1 or samples that have less timestamps than the defined lower limit
    # for L1
    if label != -1 and len(L1_ts) >= image_size:
        for i in range(num_images_L1):
            start_idx = i * image_size
            end_idx = start_idx + image_size

            # generate and scale unaugmented spectrogram from data
            spectrogram_L1 = np.zeros((len(data['freq']), image_size))
            for f_idx, freq in enumerate(data['freq']):
                spectrogram_L1[f_idx, :] = (L1_SFTs[f_idx, start_idx:end_idx] - min_dataset_val) / (max_dataset_val - min_dataset_val)

            # add unaugmented spectrogram that was just processed
            spectrograms_list_L1.append({'spectrogram': spectrogram_L1, 'label': label, 'num_images': num_images_L1})

            # apply transformation and add augmented spectrogram if requested
            if transform:
                augmented_spectrogram_L1 = apply_transform(transform, spectrogram_L1)
                spectrograms_list_L1.append({'spectrogram': augmented_spectrogram_L1, 'label': label, 'num_images': num_images_L1})

    # skip samples with a label of -1 or samples that have less timestamps than the defined lower limit
    # for H1
    if label != -1 and len(H1_ts) >= image_size:
        for i in range(num_images_H1):
            start_idx = i * image_size
            end_idx = start_idx + image_size

            # generate and scale unaugmented spectrogram from data
            spectrogram_H1 = np.zeros((len(data['freq']), image_size))
            for f_idx, freq in enumerate(data['freq']):
                spectrogram_H1[f_idx, :] = (H1_SFTs[f_idx, start_idx:end_idx] - min_dataset_val) / (max_dataset_val - min_dataset_val)

            # add unaugmented spectrogram that was just processed
            spectrograms_list_H1.append({'spectrogram': spectrogram_H1, 'label': label, 'num_images': num_images_H1})

            # apply transformation and add augmented spectrogram if requested
            if transform:
                augmented_spectrogram_H1 = apply_transform(transform, spectrogram_H1)
                spectrograms_list_H1.append({'spectrogram': augmented_spectrogram_H1, 'label': label, 'num_images': num_images_H1})

    return spectrograms_list_L1, spectrograms_list_H1

# define transformation
# horizontal translation in our case
shift = transforms.RandomAffine(degrees=0, translate=(0.1, 0))

# set the number of files for training
# keep in mind, if you apply image augmentation there will be 2 times the number of defined n_files_for_training
# ^ because for each sample, the untransformed spectrogram AND the augmented spectrogram will both be added to the training data
n_files_for_training = 30
# define an upper limit for the files
# your number of testing files will be (n_files_cap - n_files_for_training)
n_files_cap = 150
# keep in mind the above numbers are so low in order for testing this proof of concept

# initialize empty lists to store all spectrograms and labels for both observations for both training and testing
all_spectrograms_L1_train = []
all_spectrograms_H1_train = []
all_spectrograms_L1_test = []
all_spectrograms_H1_test = []

# there is a very small chance that in a sample, L1 and H1 could have different num_images
# they often have a different number of timesteps as the detectors sometimes go down for maintenance and other reasons
# store num_images for each sample in training for both observations for both training and testing
all_spectrograms_L1_train = []
num_images_list_L1_train = [] 
num_images_list_H1_train = []
num_images_list_L1_test = [] 
num_images_list_H1_test = [] 

# process files for training
for i, hdf5_file in enumerate(glob.glob(str(data_path / '*.hdf5'))[:n_files_for_training]):
    data = extract_data_from_hdf5(hdf5_file, train_labels)
    spectrograms_list_L1, spectrograms_list_H1 = create_splits(data, transform=shift)
    num_images_L1 = len(spectrograms_list_L1)
    num_images_H1 = len(spectrograms_list_H1)

    # store num_images information for each L1 and H1 sample in training
    num_images_list_L1_train.extend([num_images_L1] * num_images_L1)
    num_images_list_H1_train.extend([num_images_H1] * num_images_H1)

    # extend lists for training data
    all_spectrograms_L1_train.extend(spectrograms_list_L1)
    all_spectrograms_H1_train.extend(spectrograms_list_H1)

# create dataset and pytorch DataLoader for both H1 and L1 observations for training
train_dataset_L1 = SpectrogramDataset(spectrograms_list=all_spectrograms_L1_train, num_images_list=num_images_list_L1_train, transform=shift)
train_dataset_H1 = SpectrogramDataset(spectrograms_list=all_spectrograms_H1_train, num_images_list=num_images_list_H1_train, transform=shift)
batch_size = 32
train_dataloader_L1 = DataLoader(train_dataset_L1, batch_size=batch_size, shuffle=True)
train_dataloader_H1 = DataLoader(train_dataset_H1, batch_size=batch_size, shuffle=True)

# process remaining files for testing
for hdf5_file in glob.glob(str(data_path / '*.hdf5'))[n_files_for_training:n_files_cap]:
    data = extract_data_from_hdf5(hdf5_file, train_labels)
    spectrograms_list_L1, spectrograms_list_H1 = create_splits(data, transform=None)  # No augmentation for testing
    num_images_L1 = len(spectrograms_list_L1)
    num_images_H1 = len(spectrograms_list_H1)

    # store num_images information for each L1 and H1 sample in testing
    num_images_list_L1_test.extend([num_images_L1] * num_images_L1)
    num_images_list_H1_test.extend([num_images_H1] * num_images_H1)

    # extend lists for testing data
    all_spectrograms_L1_test.extend(spectrograms_list_L1)
    all_spectrograms_H1_test.extend(spectrograms_list_H1)

# create dataset and pytorch DataLoader for both H1 and L1 observations for testing
test_dataset_L1 = SpectrogramDataset(spectrograms_list=all_spectrograms_L1_test, num_images_list=num_images_list_L1_test)
test_dataset_H1 = SpectrogramDataset(spectrograms_list=all_spectrograms_H1_test, num_images_list=num_images_list_H1_test)

test_dataloader_L1 = DataLoader(test_dataset_L1, batch_size=batch_size, shuffle=False)
test_dataloader_H1 = DataLoader(test_dataset_H1, batch_size=batch_size, shuffle=False)

# define the DenseNet architecture
# dense block in the DenseNet architecture
class DenseNetBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseNetBlock, self).__init__()

        # reduce the number of channels before applying the densely connected layers
        # for computational efficiency
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
        )

        # feature map generation
        self.layers = nn.Sequential(
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        bottleneck_output = self.bottleneck(x)
        out = self.layers(bottleneck_output)
        # concatenate input and output along the channel dimension (dense connectivity)
        # feature maps from all preceding layers are combined in each layer within the dense block
        out = torch.cat([x, out], 1)
        return out

# improve efficiency in feature mapping - dimension reduction and compress the number of channels between dense blocks
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layers(x)

# main class for DenseNet architecture
class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=32, block_config=(6, 12, 24)):
        super(DenseNet, self).__init__()

        num_init_features = growth_rate * 2

        # initial set of layers
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # dense blocks and transition layers
        in_channels = num_init_features
        # iterate through layers in each dense block of the DenseNet
        for i, num_layers in enumerate(block_config):
            # create block
            block = self._make_dense_block(in_channels, growth_rate, num_layers)
            # add to self.features container
            self.features.add_module(f'denseblock{i + 1}', block)
            # update in_channels
            in_channels += num_layers * growth_rate

            # if this is not the last block, we need to add a transition block after it
            if i != len(block_config) - 1:
                # reduce the number of channels
                out_channels = in_channels // 2
                # create transition block
                trans = TransitionBlock(in_channels, out_channels)
                # add to container
                self.features.add_module(f'transition{i + 1}', trans)
                # update channels
                in_channels = out_channels

        # BatchNorm and ReLU before global average pooling
        self.features.add_module('norm5', nn.BatchNorm2d(in_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected layer
        self.classifier = nn.Linear(in_channels, num_classes)

    # create list of DenseNetBlock modules to return as a single nn.Sequential module
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(DenseNetBlock(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    # forward pass
    # take tensor, return output logits
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
# convert the weights and biases of specific layers to float32 data type
# data type compatibility for model
def convert_to_float32(module):
    if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.linear.Linear):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()

# number of classes (binary classification)
num_classes = 2

# number of input channels in spectrograms
num_input_channels = 3

# create instance of the DenseNet model
model = DenseNet(num_input_channels)
model.apply(convert_to_float32)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# move model to device
model.to(device)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
     # set the model to training mode
    model.train() 
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            spectrograms = inputs['spectrogram']
            labels = inputs['label']
            # convert the spectrograms and labels to appropriate data type and move to device
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            # zero gradients so fresh gradients can be computed for the current batch
            optimizer.zero_grad()
            # forward pass
            outputs = model(spectrograms)
            # compute the loss
            loss = criterion(outputs, labels)
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * spectrograms.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# train on L1 and H1
train_model(model, train_dataloader_L1, criterion, optimizer)
train_model(model, train_dataloader_H1, criterion, optimizer)

def test_model(model, dataloader):
    '''
    I explained this earlier in the create_splits function, but this is an interesting approach to classification
    for each detector, we take the num_images 360x360 spectrograms and get their soft classifications
    we then sum these soft classifications, divide by num_images, and provide a cutoff to determine the hard class
    this resulting hard class is the label for the respective detector's sample
    '''
    model.eval()  # set the model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs in dataloader:
            spectrograms = inputs['spectrogram']
            labels = inputs['label']
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            outputs = model(spectrograms)
            # get the softmax probabilities for each subsetted spectrogram in a sample
            # keep in mind, in our case we only have one spectrogram per sample based on our image_size=4096
            softmax_probs = torch.softmax(outputs, dim=1)
            # if we were to have multiple spectrograms per sample, this line would calculate the average
            # softmax probabilities for each subsetted spectrogram in a sample
            # then classify based off of that averaged softmax probability

            # append the softmax probabilities and labels to the list
            all_predictions.extend(softmax_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    # apply a cutoff value
    cutoff = 0.5
    final_predictions = np.argmax(all_predictions, axis=1)
    final_predictions[all_predictions.max(axis=1) < cutoff] = 0
    accuracy = np.mean(final_predictions == np.array(all_labels))
    print(f"\nTesting Accuracy: {accuracy:.4f}")

    return final_predictions

# test the model on L1 and H1
final_predictions_L1 = test_model(model, test_dataloader_L1, num_images_list_L1_test[n_files_for_training:n_files_cap])
final_predictions_H1 = test_model(model, test_dataloader_H1, num_images_list_H1_test[n_files_for_training:n_files_cap])
