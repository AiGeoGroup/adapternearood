import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
import shutil
from PIL import Image


def get_eurosat_templates_and_new_names():
    template = [
        'a centered satellite photo of {}.',
        'A top-down view of {} arranged in a pattern.'
    ]

    negative_template = [
        'a centered satellite photo without {}.',
        'A top-down view without {} arranged in a pattern.'
    ]

    Eurosat_NEW_CNAMES = {
        'AnnualCrop': 'Annual Crop Land',
        'Forest': 'Forest',
        'HerbaceousVegetation': 'Herbaceous Vegetation Land',
        'Highway': 'Highway or Road',
        'Industrial': 'Industrial Buildings',
        'Pasture': 'Pasture Land',
        'PermanentCrop': 'Permanent Crop Land',
        'Residential': 'Residential Buildings',
        'River': 'River',
        'SeaLake': 'Sea or Lake'
    }

    return template, negative_template, Eurosat_NEW_CNAMES


def get_train_test_val_dataset_eurosat(
        data_root='/kaggle/input/rgbeurosat/RBG/', preprocess=None):
    path_to_train_data = data_root + 'train/'
    train_dataset = torchvision.datasets.ImageFolder(path_to_train_data,
                                                     transform=preprocess)

    path_to_test_data = data_root + 'test/'
    test_dataset = torchvision.datasets.ImageFolder(path_to_test_data,
                                                    transform=preprocess)

    path_to_val_data = data_root + 'val/'
    val_dataset = torchvision.datasets.ImageFolder(path_to_val_data,
                                                   transform=preprocess)
    return train_dataset, test_dataset, val_dataset


def get_train_test_val_dataset_split(
        data_root='/kaggle/input/nwpu-data-set/NWPU Data Set/NWPU-RESISC45/NWPU-RESISC45/',
        preprocess=None,
        tain_size=0.8,
        test_size=0.1,
        val_size=0.1):
    full_dataset = torchvision.datasets.ImageFolder(data_root,
                                                    transform=preprocess)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [tain_size, test_size, val_size])
    return full_dataset, train_dataset, val_dataset, test_dataset


def get_train_test_val_dataset_AID(
        data_root='/kaggle/working/aid-scene-classification-datasets/AID/', preprocess=None):
    path_to_train_data = data_root + 'train/'
    train_dataset = torchvision.datasets.ImageFolder(path_to_train_data,
                                                     transform=preprocess)

    path_to_test_data = data_root + 'test/'
    test_dataset = torchvision.datasets.ImageFolder(path_to_test_data,
                                                    transform=preprocess)

    path_to_val_data = data_root + 'val/'
    val_dataset = torchvision.datasets.ImageFolder(path_to_val_data,
                                                   transform=preprocess)
    return train_dataset, test_dataset, val_dataset


def get_train_test_val_dataset_siri_whu(
        data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/', preprocess=None):
    path_to_train_data = data_root + 'train/'
    train_dataset = torchvision.datasets.ImageFolder(path_to_train_data,
                                                     transform=preprocess)

    path_to_test_data = data_root + 'test/'
    test_dataset = torchvision.datasets.ImageFolder(path_to_test_data,
                                                    transform=preprocess)

    path_to_val_data = data_root + 'val/'
    val_dataset = torchvision.datasets.ImageFolder(path_to_val_data,
                                                   transform=preprocess)
    return train_dataset, test_dataset, val_dataset

def get_train_test_val_dataset_resisc(
        data_root='/kaggle/input/nwpuresisc45/Dataset/', preprocess=None):
    path_to_train_data = data_root + 'train/train/'
    train_dataset = torchvision.datasets.ImageFolder(path_to_train_data,
                                                     transform=preprocess)

    path_to_test_data = data_root + 'test/test/'
    test_dataset = torchvision.datasets.ImageFolder(path_to_test_data,
                                                    transform=preprocess)
    return train_dataset, test_dataset, test_dataset


def get_train_test_val_dataset_UCMerced(
        data_root='/kaggle/input/landuse-scene-classification/images_train_test_val/',
        preprocess=None):
    path_to_train_data = data_root + 'train/'
    train_dataset = torchvision.datasets.ImageFolder(path_to_train_data,
                                                     transform=preprocess)

    path_to_test_data = data_root + 'test/'
    test_dataset = torchvision.datasets.ImageFolder(path_to_test_data,
                                                    transform=preprocess)

    path_to_test_data = data_root + 'validation/'
    val_dataset = torchvision.datasets.ImageFolder(path_to_test_data,
                                                    transform=preprocess)
    return train_dataset, val_dataset, test_dataset

class Datum:
    """Data instance which defines the basic attributes.
    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = ''  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    def get_num_classes(self, data_source):
        """Count number of classes.
        Args: data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).
        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError('Input domain must belong to {}, '
                                 'but got [{}]'.format(self.domains, domain))

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(self,
                                 *data_sources,
                                 num_shots=-1,
                                 repeat=True):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(TorchDataset):

    def __init__(self,
                 data_source,
                 input_size,
                 transform=None,
                 is_train=False,
                 return_img0=False,
                 k_tfm=1):
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError('Cannot augment the image {} times '
                             'because transform is None'.format(self.k_tfm))

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_data_loader(data_source=None,
                      batch_size=64,
                      input_size=224,
                      tfm=None,
                      is_train=True,
                      shuffle=False,
                      dataset_wrapper=None):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source,
                        input_size=input_size,
                        transform=tfm,
                        is_train=is_train),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()))
    assert len(data_loader) > 0

    return data_loader



# 创建few shot数据集合
def generate_fewshot_dataset(classnames, train_data, num_shots=1, repeat=True):
    each_label_data_len = np.array([0 for x in range(len(classnames))])
    check_len = np.array([num_shots for x in range(len(classnames))])

    # few_shot_train = []
    few_shot_image_train = []
    few_shot_label_train = []

    for x in train_data:
        if each_label_data_len[x[1]] < num_shots:
            each_label_data_len[x[1]] = each_label_data_len[x[1]] + 1
            # few_shot_train.append(x)
            few_shot_image_train.append(x[0])
            few_shot_label_train.append(x[1])
        # print(type(x[0]))
        # print(type(x[1]))
        # print(x[1])

        if ((each_label_data_len == check_len).all()): break
        # break

    tensor_x = torch.Tensor(
        np.array(few_shot_image_train))  # transform to torch tensor
    tensor_y = torch.Tensor(np.array(few_shot_label_train).astype(np.int64))

    few_shot_dataset = TensorDataset(tensor_x,
                                     tensor_y)  # create few shot datset

    return few_shot_dataset


class SatellitedBase(DatasetBase):

    def __init__(self,
                 configs,
                 num_shots=1,
                 classnames=None,
                 sat_name='Eurosat'):
        root = configs['data_root']
        self.cupl_path = configs['cupl_path']
        self.template, self.negative_template, eurosat_class_names = get_eurosat_templates_and_new_names()
        if sat_name == 'Eurosat':
            self.classnames = self.update_classname_with_old(
                classnames, eurosat_class_names)
            # train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            # super().__init__(train_x=train, val=val, test=test)
        else:
            self.classnames = classnames

    def update_classname_with_old(self, classnames, NEW_CNAMES):
        new_class_names = []
        for classname in classnames:
            new_class_names.append(NEW_CNAMES[classname])
        return new_class_names

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CNAMES[cname_old]
            item_new = Datum(impath=item_old.impath,
                             label=item_old.label,
                             classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new


'''From SuS-X'''


def get_num_classes_from_dsname(dataset_name):
    if (dataset_name == 'imagenet'):
        return 1000
    elif (dataset_name == 'imagenet-sketch'):
        return 1000
    elif (dataset_name == 'imagenet-r'):
        return 200
    elif (dataset_name == 'stanfordcars'):
        return 196
    elif (dataset_name == 'ucf101'):
        return 101
    elif (dataset_name == 'country211'):
        return 211
    elif (dataset_name == 'birdsnap'):
        return 500
    elif (dataset_name == 'caltech101'):
        # from CoOP paper:
        #
        # For Caltech101, the "BACKGROUND Google"
        # and "Faces easy" classes are discarded
        return 100
    elif (dataset_name == 'caltech256'):
        return 257
    elif (dataset_name == 'flowers102'):
        return 102
    elif (dataset_name == 'cub'):
        return 200
    elif (dataset_name == 'sun397'):
        return 397
    elif (dataset_name == 'dtd'):
        return 47
    elif (dataset_name == 'fgvcaircraft'):
        return 100
    elif (dataset_name == 'oxfordpets'):
        return 37
    elif (dataset_name == 'food101'):
        return 101
    elif (dataset_name == 'cifar10'):
        return 10
    elif (dataset_name == 'cifar100'):
        return 100

    elif (dataset_name == 'Eurosat'):
        return 10

    elif (dataset_name == 'NWPU-RESISC45'):
        return 45

    elif (dataset_name == 'UC-Merced'):
        return 21

    elif (dataset_name == 'siri-wuhu'):
        return 12

    elif (dataset_name == 'AID-Scene'):
        return 30


def get_num_classes(dataset):
    if dataset == None: return 0
    else: return len(dataset.classes)

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def make_aid_train_test_dataset(
    data_path = '/kaggle/input/aid-scene-classification-datasets/AID/',
    dest_path = '/kaggle/working/aid-scene-classification-datasets/AID/'):
    

    # path to destination folders
    train_folder = os.path.join(dest_path, 'train')
    val_folder   = os.path.join(dest_path, 'val')
    test_folder  = os.path.join(dest_path, 'test')


    # Define a list of image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'tif']

    # Create a list of image filenames in 'data_path'
    imgs_list = []

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.split(".")[1] in image_extensions:
                imgs_list.append(os.path.join(dirname, filename))

    random.seed(42) # Sets the random seed 
    random.shuffle(imgs_list) # Shuffle the list of image filenames

    # determine the number of images for each set
    train_size = int(len(imgs_list) * 0.70)
    val_size = int(len(imgs_list) * 0.15)
    test_size = int(len(imgs_list) * 0.15)

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + val_size:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        f_dir = f.split('/')[-2] + '/'

        f_dir = os.path.join(dest_folder, f_dir)
        if not os.path.exists(f_dir): os.makedirs(f_dir)
        shutil.copy(f, os.path.join(dest_folder, f_dir))
