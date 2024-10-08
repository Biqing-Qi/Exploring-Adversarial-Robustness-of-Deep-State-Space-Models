from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from typing import Any
from torch.utils.data import Subset
import random
from collections import defaultdict
from torchvision import transforms
from models.SSM import SSM, Mega, S5_SSM, S6_SSM

class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(root + "/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
def filter_dataset_by_random_classes(dataset, k):
    """
    Randomly select k classes from the dataset and filter out the samples that only belong to these classes.

    Params:
    - dataset: The input dataset.
    - k: The number of classes to randomly select.

    Return:
    - class_indices_map: A dictionary containing the sample indices for each class.
    """
    # 收集所有类别及其对应的样本索引
    all_classes = sorted(set([label for _, label in dataset]))
    random_classes = random.sample(all_classes, k)
    class_indices_map = defaultdict(list)
    
    for idx, (_, label) in enumerate(dataset):
        if label in random_classes:
            class_indices_map[label].append(idx)
    
    return class_indices_map, random_classes

def create_subsets(dataset, class_indices_map):
    """
    Create a new subset dataset using the given class index mapping.

    Params:
    - dataset: The input dataset.
    - class_indices_map: A dictionary containing the classes to filter and their corresponding sample index lists.

    Return:
    - subsets: A dictionary containing the new subset datasets.
    """
    subsets = {}
    for class_, indices in class_indices_map.items():
        subset = Subset(dataset, indices)
        subsets[class_] = subset
    return subsets

def load_tinyimagenet(args):
    batch_size = args.batch_size
    nw = args.workers
    root = '/mnt/petrelfs/qibiqing/datasets/tiny-imagenet-200'
    id_dic = {}
    for i, line in enumerate(open(root+'/wnids.txt','r')):
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    data_transform = {
        "train": transforms.Compose([transforms.RandomCrop(64, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()
                                     ]),
        "val": transforms.Compose([transforms.Resize(64),
                                   transforms.ToTensor()])}
    train_dataset = TrainTinyImageNet(root, id=id_dic, transform=data_transform["train"])
    val_dataset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])
    selected_class_indices = list(range(args.num_classes))
    new_trainset_indices = [index for index, item in enumerate(train_dataset) if item[1] in selected_class_indices]
    new_testset_indices = [index for index, item in enumerate(val_dataset) if item[1] in selected_class_indices]
    num_classes = args.num_classes
    train_dataset = Subset(train_dataset, new_trainset_indices)
    val_dataset = Subset(val_dataset, new_testset_indices)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=nw)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw)
    
    print("TinyImageNet Loading SUCCESS"+
          "\nlen of train dataset: "+str(len(train_dataset))+
          "\nlen of val dataset: "+str(len(val_dataset)))
    
    return train_loader, val_loader, num_classes

def filter_dataset_by_class(dataset, num_samples_per_class):
    """
    Randomly select a specified number of samples from each class and return a new dataset.

    Params:
    - dataset: The input dataset.
    - num_samples_per_class: The number of samples to select from each class.

    Return:
    - filtered_indices: A list containing the indices of the randomly selected samples.
    """
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # 从每个类别中随机选择指定数量的样本索引
    filtered_indices = []
    for label, indices in class_indices.items():
        sampled_indices = random.sample(indices, num_samples_per_class)
        filtered_indices.extend(sampled_indices)
    filtered_dataset = Subset(dataset, filtered_indices)

    return filtered_dataset

def build_model(args, model_name):
    if model_name == 'SSM':
        if args.use_AdSS:
            model = SSM(d_input=3, d_model=128, n_layers=args.num_layers, \
                use_AdSS=True,AdSS_Type=args.AdSS_Type, d_output=args.num_classes )
        else:
            model = SSM(d_input=3, d_model=128, n_layers=args.num_layers, d_output=args.num_classes )
    elif model_name == 'DSS':
        if args.use_AdSS:
            model = SSM(d_input=3, d_model=128, n_layers=args.num_layers, mode = 'diag', \
                use_AdSS=True,AdSS_Type=args.AdSS_Type, d_output=args.num_classes )
        else:
            model = SSM(d_input=3, d_model=128, n_layers=args.num_layers, mode = 'diag', d_output=args.num_classes )
    elif model_name == 'S5':
        model = S5_SSM(d_input=3, d_model=128, n_layers=args.num_layers, d_output=args.num_classes )
    elif model_name == 'Mega':
        model = Mega(d_input=3, d_model=128, n_layers=args.num_layers, d_output=args.num_classes, seq_len=64 * 64 ) 
    elif model_name == 'S6':
        model = S6_SSM(d_input=3, d_model=128, n_layers=args.num_layers, d_output=args.num_classes )     
    return model