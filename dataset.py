import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
import h5py
import numpy as np
from PIL import Image
import pickle
import time
import matplotlib.pyplot as plt

#%%
# Define paths
hdf5_folder = "/mnt/mdpm/d01/sftp/jilek/data/02/export_100_downsample_1_noisy/"
output_folder = "//mnt/mdpm/d01/sftp/jilek/data/02/export_100_downsample_1_noisy_diffpats"





# HDF5 file path
file_path = "/mnt/mdpm/d01/sftp/jilek/data/02/export_100_downsample_2_noisy/00000001.hdf5"
# Open the HDF5 file in read mode
with h5py.File(file_path, "r") as f:
    # Display all keys within the HDF5 file
    print("Keys within the HDF5 file:")
    print("---------------------------")
    for key in f.keys():
        print(key)

# Open the HDF5 file in read mode
with h5py.File(file_path, "r") as f:
    # Check if 'mask' key exists in the HDF5 file

        # Access the mask dataset
        mask_data = f['phase'][:]

        # Display the mask using imshow
        plt.imshow(mask_data, )
        plt.title('Mask')
        plt.colorbar(label='Intensity')
        plt.show()










# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)



start = time.time()

# Counter for tracking the number of iterations
iteration_count = 0
# A few seconds later
# Iterate through HDF5 files in the folder
for filename in os.listdir(hdf5_folder):
    if filename.endswith(".hdf5"):
        hdf5_path = os.path.join(hdf5_folder, filename)


        # Load data from HDF5 file
        with h5py.File(hdf5_path, 'r') as file:
            # Extract 'diffpats' dataset
            if 'diffpats' in file:
                diffpats = file['diffpats'][()]
            else:
                continue  # Skip this file if 'diffpats' dataset is not found

        # Save the diffpats array as a pickle file
        output_filename = os.path.splitext(filename)[0] + '.npy'
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(diffpats, f)


# Calculate the end time and time taken
end = time.time()
length = start - end

# Show the results : this can be altered however you like
print("It took", start-end, "seconds!")





npy_file_path = "/mnt/mdpm/d01/sftp/jilek/data/02/export_100_downsample_1_noisy_diffpats/00079087.npy"

# Load the array from the .npy file
array = np.load(npy_file_path,allow_pickle=True)
# Visualize the array using imshow
plt.imshow(array[0,:,:])

plt.title('Visualization of NumPy Array')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()



#%%
class CustomDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.input_files = sorted(os.listdir(input_folder))
        self.target_files = sorted(os.listdir(target_folder))
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):

        input_folder_segments = self.input_folder.split('/')[-3:]
        input_file = os.path.join('/'.join(input_folder_segments), self.input_files[idx])
        target_folder_segments = self.target_folder.split('/')[-3:]
        target_file = os.path.join('/'.join(target_folder_segments), self.target_files[idx])
        return input_file, target_file




# Transformation to apply to the images
# Define input and target folders
input_folder = "/mnt/mdpm/d01/sftp/jilek/data/02/export_100_downsample_1_noisy_diffpats/export_100_downsample_1_noisy_diffpats/"
target_folder = "/mnt/mdpm/d01/sftp/jilek/data/01/orig/"



dataset = CustomDataset(input_folder, target_folder)



# Create JSON file with file paths
json_data = []

for i in range(len(dataset)):
    input_path, target_path = dataset[i]
    json_data.append({
        "diffpatts": input_path,
        "target": target_path
    })




json_file_path = "dataset_info.json"
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("Dataset JSON file created:", json_file_path)


#%%


# Create DataLoader
batch_size = 32  # Adjust as needed
shuffle = True   # Set to True if you want to shuffle the data
num_workers = 4  # Number of subprocesses to use for data loading
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


for batch_idx, (input_data, target_data) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print("Input data paths:")
    for input_path in input_data:
        print(input_path)
    print("Target data paths:")
    for target_path in target_data:
        print(target_path)
    print()




# Create JSON file with file paths
json_data = []

for batch_idx, (input_data, target_data) in enumerate(dataloader):
    for input_path, target_path in zip(input_data, target_data):
        json_data.append({
            "diffpatts": input_path,
            "target": target_path
        })



































class CustomDataset(Dataset):
    def __init__(self, hdf5_folder="data/01/export_100_downsample_1_noisy/", label_folder="data/01/orig/",
                 transform=None):
        # Get the current working directory
        cwd = os.getcwd()

        # Construct absolute paths using relative paths
        self.hdf5_folder = os.path.join(cwd, hdf5_folder)
        self.label_folder = os.path.join(cwd, label_folder)

        # List files in the directories
        self.hdf5_files = [f for f in os.listdir(self.hdf5_folder) if f.endswith('.h5') or f.endswith('.hdf5')]

        self.transform = transform

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, idx):
        hdf5_file = self.hdf5_files[idx]
        hdf5_path = os.path.join(self.hdf5_folder, hdf5_file)

        # Load data from HDF5 file
        with h5py.File(hdf5_path, 'r') as file:
            dataset_name = list(file.keys())[0]
            data = file[dataset_name][()]

        # Load label image
        label_file = hdf5_file.split('.')[0] + '.jpg'  # Assuming label image file extension is '.jpg'
        label_path = os.path.join(self.label_folder, label_file)
        label_image = Image.open(label_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            data = self.transform(data)
            label_image = self.transform(label_image)

        return data, label_image

custom_dataset = CustomDataset()



#%%
class CustomDataset(Dataset):
    def __init__(self, hdf5_folder, label_folder, transform=None):
        #self.hdf5_files = [f for f in os.listdir(hdf5_folder) if f.endswith('.h5') or f.endswith('.hdf5')]
        self.hdf5_folder = hdf5_folder
        self.hdf5_file =os.listdir(hdf5_folder)
        self.image_list = os.listdir( label_folder)
        self.label_folder = label_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
       # hdf5_file = self.hdf5_files[idx]
        hdf5_path = os.path.join(self.hdf5_folder, hdf5_file)

        # Load data from HDF5 file
        with h5py.File(hdf5_path, 'r') as file:
            mask = file['diffpats'][()]  # Extract 'diffpats' dataset
        img_name = os.path.join(self.label_folder, self.image_list[idx])
        label_image = Image.open(img_name)
        # Load label image

        return mask, label_image


# Define transformations to apply to data and labels
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations as needed
])

# Define paths to HDF5 folder and label image folder
hdf5_folder = "/mnt/mdpm/d01/sftp/jilek/data/01/export_100_downsample_1_noisy/"
label_folder = "/mnt/mdpm/d01/sftp/jilek/data/01/orig/"

# Create an instance of CustomDataset
custom_dataset = CustomDataset(hdf5_folder, label_folder)

batch_size = 32

# Create a DataLoader for the custom dataset
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches in the DataLoader
for batch in data_loader:
    # Extract data and labels from the batch
    data, label_image = batch
    break


class CustomDataset(Dataset):
    def __init__(self, hdf5_folder, label_folder, transform=None):
        self.hdf5_files = [f for f in os.listdir(hdf5_folder) if f.endswith('.h5') or f.endswith('.hdf5')]
        self.label_folder = label_folder
        self.transform = transform

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, idx):
        hdf5_file = self.hdf5_files[idx]
        hdf5_path = os.path.join(hdf5_folder, hdf5_file)

        # Load data from HDF5 file
        with h5py.File(hdf5_path, 'r') as file:
            dataset_name = list(file.keys())[0]
            data = file[dataset_name][()]

        # Load label image
        label_file = hdf5_file.split('.')[0] + '.jpg'
        label_path = os.path.join(label_folder, label_file)
        label_image = Image.open(label_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            data = self.transform(data)
            label_image = self.transform(label_image)

        return data, label_image


# Define transformations to apply to data and labels
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations as needed
])

# Define paths to HDF5 folder and label image folder
hdf5_folder =  "/mnt/mdpm/d01/sftp/jilek/data/01/export_100_downsample_1_noisy/"
label_folder = "/mnt/mdpm/d01/sftp/jilek/data/01/orig"

# Create an instance of CustomDataset
custom_dataset = CustomDataset(hdf5_folder, label_folder, transform=transform)

batch_size = 32

# Create an instance of CustomDataset


# Create a DataLoader for the custom dataset
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the DataLoader
for batch_data, batch_labels in data_loader:
    # Perform operations on the batched data and labels
    # Here, batch_data is a tensor containing a batch of data
    # and batch_labels is a tensor containing a batch of labels

    # Example operation: Print the shapes of batched data and labels
    print("Batch data shape:", batch_data.shape)
    print("Batch labels shape:", batch_labels.shape)





class HDF5Dataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.h5') or f.endswith('.hdf5')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.folder_path, file_name)
        with h5py.File(file_path, "r") as file:
            # Assuming there's only one dataset in the HDF5 file
            dataset_name = list(file.keys())[0]
            dataset = file[dataset_name]
            data = dataset[()]
            return torch.tensor(data)





# Path to your HDF5 file
folder_path = "/mnt/mdpm/d01/sftp/jilek/data/01/export_100_downsample_1_noisy/"

# Create an instance of the HDF5Dataset class
dataset = HDF5Dataset(folder_path)

# Access an item from the dataset
sample_data = dataset[0]

# Get the length of the dataset
dataset_length = len(dataset)

# Print information
print("Shape of sample data:", sample_data.shape)
print("Length of the dataset:", dataset_length)









class CustomDataset(Dataset):
    def __init__(self, hdf5file, jpg_dir, transform=None):
        self.hdf5file = hdf5file
        self.jpg_dir = jpg_dir
        self.transform = transform

        with h5py.File(self.hdf5file, 'r') as db:
            self.keys = list(db.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5file, 'r') as db:
            key = self.keys[idx]
            image = db[key][:]  # Read data using the key from the HDF5 file

        label_path = os.path.join(self.jpg_dir, f"label_{key}.jpg")  # Assuming label files are named like 'label_0.jpg', 'label_1.jpg', etc.
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# Create instances of your dataset and dataloader
dataset = CustomDataset(img_dir="/mnt/mdpm/d01/sftp/jilek/data/01/export_100_downsample_1_noisy/", label_dir="/mnt/mdpm/d01/sftp/jilek/data/01/orig/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Assuming you have already created a DataLoader named 'data_loader'
# Iterate through the DataLoader to access a batch of data
for images, labels in data_loader:
    # Display the first image and label in the batch
    img = images[0].squeeze().numpy()  # Convert the tensor to a numpy array and remove single-dimensional entries
    label = labels[0].item()  # Extract the label value

    # Display the image and label
    plt.imshow(img, cmap="gray")  # Display the image in grayscale
    plt.title(f"Label: {label}")  # Set the title of the plot to show the label
    plt.axis('off')  # Turn off axis for cleaner display
    plt.show()  # Show the image with its corresponding label

    break




















































class Deep_Stem_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = os.listdir(img_dir)
        self.label_list = os.listdir(label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        label_name = self.label_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Load the image
        image = read_image(img_path)

        # Load the label
        label = read_image(label_path)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor()  # Convert image to tensor
])



# Specify directory containing images
img_dir = "/mnt/mdpm/d01/sftp/jilek/data/01/export_100_downsample_1_noisy/"

label_dir = "/mnt/mdpm/d01/sftp/jilek/data/01/orig/"

# Create instance of CustomImageDataset
custom_dataset = Deep_Stem_Dataset(img_dir=img_dir, label_dir=label_dir, transform=transform)

# Create DataLoader object
data_loader = DataLoader(custom_dataset , batch_size=32, shuffle=True)








lass H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        # Do something with file and return data

    def __len__(self):
        with h5py.File(self.h5_path,'r') as file:
            return len(file["dataset"])


class Deep_Stem_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = os.listdir(img_dir)
        self.label_list = os.listdir(label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        label_name = self.label_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Load the image from HDF5 file
        with h5py.File(img_path, 'r') as img_file:
            image = img_file['data'][()]  # Assuming 'data' is the dataset name

        if self.transform:
            # Assuming transform is applied to each channel independently
            for i in range(image.shape[0]):
                image[i] = self.transform(image[i])
            label = self.transform(label)

        return image, label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor()  # Convert image to tensor
])

# Specify directory containing images
img_dir = "/mnt/mdpm/d01/sftp/jilek/data/01/export_100_downsample_1_noisy/"
label_dir = "/mnt/mdpm/d01/sftp/jilek/data/01/orig/"

# Create instance of Deep_Stem_Dataset
custom_dataset = Deep_Stem_Dataset(img_dir=img_dir, label_dir=label_dir, transform=transform)

# Create DataLoader object
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)



# Display image and label.
train_features, train_labels = next(iter(data_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Assuming you're working with grayscale images
img = train_features[0].squeeze().numpy()
label = train_labels[0]

plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")



