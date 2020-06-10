import os
import pickle
import sqlite3
from io import StringIO

import cv2
import lmdb
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import trange

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


####################
# Files & IO
####################


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')

    if os.path.isfile(keys_cache_file):
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii')
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


class ImageDataset(Dataset):
    '''Read images'''

    def __init__(self, path, transform):
        super(ImageDataset, self).__init__()

        self.transform = transform
        self.paths = None
        self.env = None  # environment for lmdb

        # read image list from lmdb or image files
        self.env, self.paths = get_image_paths('lmdb', path)
        assert self.paths, 'Error: paths are empty.'

    def __getitem__(self, index):
        path = None

        # get LR image
        path = self.paths[index]
        img = read_img(self.env, path)
        H, W, C = img.shape

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        # img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.paths)


class SCGN_Dataset(Dataset):
    def __init__(self, image_data, nz, path):
        self.image_data = image_data
        self.nz = nz
        self.path = path
        if not os.path.exists(path):
            self.connection, self.cursor = self.init()
        else:
            self.connection = sqlite3.connect(self.path)
            self.cursor = self.connection.cursor()

    def __getitem__(self, index):
        try:
            image, Y = self.image_data[index]
        except:
            image = self.image_data[index]

        Z_data = self.cursor.execute('SELECT arr FROM z WHERE idx=(?)', (index,))
        buffer = StringIO(Z_data.fetchone()[0])
        Z_data = np.loadtxt(buffer, dtype=np.float32, delimiter=',')
        return (image, Z_data, index)

    def get_Y(self, index):
        image, Y = self.image_data[index]
        return Y

    def __len__(self):
        return len(self.image_data)

    def save(self, Z_data, index):
        Z_data = Z_data.cpu().numpy()

        def sql():
            for z, i in zip(Z_data, index):
                buffer = StringIO()
                np.savetxt(buffer, z, delimiter=',')
                yield buffer.getvalue(), i.item()

        self.cursor.executemany('UPDATE z SET arr=(?) WHERE idx=(?)', sql())
        # self.cursor.execute('UPDATE z SET arr=(?) WHERE idx=(?)', (buffer.getvalue(), i.item()))

    def commit(self):
        self.connection.commit()

    def init(self):
        connection = sqlite3.connect(self.path)  # create db file
        cursor = connection.cursor()

        # create table
        cursor.execute('CREATE TABLE z (idx INTEGER PRIMARY KEY, arr TEXT)')

        n = len(self.image_data)

        def sql():
            for idx in trange(n):
                z = np.empty((1, self.nz), dtype=np.float32)
                z[:] = np.random.randn(*z.shape)  # * 0.01  # mean = 0, std = 0.01
                z_state = np.zeros((2, self.nz), dtype=np.float32)

                data = np.concatenate((z, z_state), axis=0)
                buffer = StringIO()
                np.savetxt(buffer, data, delimiter=',')

                yield idx, buffer.getvalue()

            # cursor.execute('INSERT INTO z VALUES (?,?)', (idx, buffer.getvalue()))

        cursor.executemany('INSERT INTO z VALUES (?,?)', sql())
        connection.commit()

        return connection, cursor


def load_dataset(opt):
    transform = []
    if opt['dataset'] in ['mnist']:
        transform.append(transforms.Resize((32, 32)))
    transform.append(transforms.ToTensor())
    if opt['dataset'] in ['mnist']:
        transform.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)

    if opt['dataset'] in ['mnist']:
        data = datasets.MNIST(opt['data_dir'], train=True, download=True, transform=transform)
    elif opt['dataset'] in ['cifar10']:
        data = datasets.CIFAR10(opt['data_dir'], train=True, download=True, transform=transform)
    elif opt['dataset'] in ['cifar100']:
        data = datasets.CIFAR100(opt['data_dir'], train=True, download=True, transform=transform)
    elif opt['dataset'] in ['celeba']:
        data = ImageDataset(opt['data_dir'], transform=transform)

    size = data[0][0].size()
    opt['nc'] = size[0]
    opt['im_size'] = size[1]

    train_data = data
    train_loader = DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)

    return train_data, train_loader
