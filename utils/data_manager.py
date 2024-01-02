import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, TinyImageNet200
import torch, copy
import os, pdb, random
import numpy as np
import torch.backends.cudnn as cudnn



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx, image, label = self.dataset[self.idxs[item]]
        return idx, image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(y_train, beta=0.4, n_parties=5):
    """_summary_
        按照迪利克雷分布进行数据集划分
    Args:
        y_train (_type_): 训练数据集标签列表
        beta (float, optional): . 迪利克雷概率分布参数 Defaults to 0.4.
        n_parties (int, optional): . 客户端数量 Defaults to 5.

    Returns:
        dict: client id 与对应数据索引集合的映射: 
    """

    data_size = y_train.shape[0]
    # beta = 0 表示数据集独立同分布，只是shuffle + partition
    if beta == 0:   # for iid
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # beta > 0 表示数据集非独立同分布，
    elif beta > 0:  # for niid
        min_size = 0
        min_require_size = 1
        # label = np.unique(y_train).shape[0]
        labels = np.unique(y_train)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in labels:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map


class DataManager(object):
    def __init__(self, dataset_name:str, shuffle:bool, seed, init_cls, increment):
        """_summary_
            _setup_data 下载数据，划分训练集、测试集，打乱数据
        Args:
            dataset_name (str): 数据集名称
            shuffle (bool): 是否打乱顺序
            seed (_type_): 种子
            init_cls (_type_): 每个任务的类别数量 num_classed / task_num
            increment (_type_): init_cls
        """
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        # 保证每个任务至少一个类别
        assert init_cls <= len(self._class_order), "No enough classes."
        # self._increments 每个任务类别数量的数组
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        """_summary_
            返回task数量,将nb_tasks函数转化为 名称为nb_tasks的只读属性
        Returns:
            int: task 数量
        """
        return len(self._increments)

    def get_task_size(self, task):
        """_summary_
            返回 id 为 task的类别数量
        Args:
            task (int): task id
        Returns:
            int: 该task的类别数量
        """
        return self._increments[task]

    def get_total_classnum(self):
        """_summary_
            返回总的类别数量
        Returns:
            int: 总类别数量
        """
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        """_summary_

        Args:
            indices (_type_): 
            source (str): "train" or "test" , 数据集类别
            mode (str): "train" or "test" or "flip", 数据集处理(Transforms)方法
            appendent (_type_, optional): . Defaults to None.
            ret_data (bool, optional): . Defaults to False.
            m_rate (_type_, optional): . to None.
        
        Returns:
            _type_: _description_
        """

        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # 
        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path,self._classes[indices[0]:indices[-1] + 1])
        else:
            return DummyDataset(data, targets, trsf, self.use_path, self._classes[indices[0]:indices[-1] + 1])

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data  获得训练数据集、标签集，测试数据集、标签集
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self._classes = idata.classes

        # Transforms 获得数据集Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order 设置增量学习顺序， order是从0开始的list，如果设置shuffle，就乱序，否则按idata定义的顺序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        # print(self._class_order)

        # Map indices 映射标签和类别顺序(打乱标签顺序)
        # 将标签映射成标签在order中的下标 标签 -> 下标
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        """_summary_

        Args:
            x (_type_): self._train_data 训练数据集
            y (_type_): self._train_targets 训练标签集合
            low_range (int): 
            high_range (int): 
        Returns:
            _type_: _description_
        """
        # np.logical_and 找出在low_range和high_range区间内的 index, 符合条件的为 True
        # np.where  True所在的索引
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, classes = None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    """
        返回标签集合元素在顺序序列中的对应下标
    Args:
        y (np.array): dataset.targets(数据集对应的标签集合)
        order (np.array): 数据类别顺序
    Returns:
        np.array: 标签集合在类别序列中的位置
        例如标签集合 [1,3,5,3,2]
        类别序列 [0, 1, 2, 3, 4]
        返回标签 [1, 3 ,5, 3, 2]
    """
    return np.array(list(map(lambda x: order.index(x), y)))


# idata类是类别管理器，用来划分任务，生成非独立同分布数据集
def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "tiny_imagenet":
        return TinyImageNet200()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# def accimage_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
#     accimage is available on conda-forge.
#     """
#     import accimage

#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     """
#     from torchvision import get_image_backend

#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)

