import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# def get_transform(input_size=224):
#     return transforms.Compose([
#         transforms.Resize((input_size, input_size)),
#         transforms.ToTensor(),
#         get_normalize(),
#     ])

# def get_normalize():
#     return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def PlacesDataset(Dataset):
    def __init__(self, dataset_npy:str, transform = None, onlylabels=None):
        self.data = []
        self.classes = {
            'bathroom': 0,
            'bedroom': 1,
            'childs_room': 2,
            'classroom': 3,
            'dressing_room': 4,
            'living_room': 5,
            'studio': 6,
            'swimming_pool': 7
        }
        self.onlylabels = onlylabels
        self.transform = transform
        reader = np.load(dataset_npy)
        for [img_path, label] in reader:
            self.data.append((img_path, self.classes[label])) #for some reason, label is str
        self.data = np.asarray(self.data) # data = [['fullpath', 'label'], ....]
        

        if self.onlylabels is not None:
            self.onlylabels = [int(i) for i in self.onlylabels]
            clip_indexes = np.where(self.data[:, 1] == str(self.onlylabels[0]))[0] # indexes
            for i in self.onlylabels[1:]:
                clip = np.where(self.data[:, 1] == str(i))[0]
                clip_indexes = np.append(clip_indexes, clip)
            clip_indexes.sort()
            self.data = self.data[clip_indexes]
        labels, counts = np.unique(self.data[:, 1], return_counts = True)
        labels = labels.astype(int) # labels are integers folowwing self.classes
        

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(zip(labels, counts))
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[int(cls)] for cls in self.data[:, 1]]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(self.labels)))
        for idx in self.class_counts.keys():
            print("    Class '{}' ({}): {} images.".format(
                self.classes[idx], idx, self.class_counts[idx]))

    def __getitem__(self, index:int):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.samples)
    