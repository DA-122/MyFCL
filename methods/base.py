import copy
import logging
import numpy as np
import torch
from torch import nn
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.data_manager import DummyDataset, _get_idata
from torchvision import transforms
from einops import rearrange


EPSILON = 1e-8
batch_size = 64

# 增量学习基类

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        # 标签缓冲区和数据缓冲区
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5
        self.args = args
        self.each_task = args["increment"]
        self.seed = args["seed"]
        self.tasks = args["tasks"]
        self.wandb = args["wandb"]
        self.save_dir = args["save_dir"]
        self.dataset_name = args["dataset"]
        self.nums = args["nums"]
    
        # ----
        args["memory_size"] = 2000
        args["memory_per_class"] = 20
        args["fixed_memory"] = False

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = "0"
        # self._multiple_gpus = args["device"]
        # 用来展示数据集
        # self.show_example = []

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
            
    # 需要重写的方法
    # 抽象方法：开始下一个任务
    def after_task(self):
        pass

    # 抽象方法: 增量方式进行训练
    def incremental_train(self):
        pass

    # 抽象方法: 模拟联邦学习训练
    def _fl_train(self):
        pass

    # 抽象方法：构造rehearsal_memory(自己写)
    def real_build_rehearsal_memory(self):
        pass
    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:  # false
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=self.each_task)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy


    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def combine_dataset(self, pre_dataset, cur_dataset, size):
        """合并两个数据集

        Args:
            pre_dataset (Dataset): 旧任务的Dataset
            cur_dataset (Dataset): 当前任务Dataset
            size (int): 旧任务size

        Returns:
            Dataset: 合并后的Dataset
        """
        # correct
        idx = pre_dataset.idxs
        pre_labels = pre_dataset.dataset.labels[idx]  # label 22, wrong
        pre_data = pre_dataset.dataset.images[idx]

        idx = cur_dataset.idxs
        cur_labels = cur_dataset.dataset.labels[idx]
        cur_data = cur_dataset.dataset.images[idx]

        if size !=0:
            idxs = np.random.choice(range(len(pre_dataset.idxs)), size, replace=False)
            selected_exemplar_data, selected_exemplar_label = pre_data[idxs], pre_labels[idxs]
            
            combined_data = np.concatenate((cur_data, selected_exemplar_data),axis=0)
            combined_label = np.concatenate((cur_labels, selected_exemplar_label),axis=0)
            # combined_label = np.concatenate(combined_label)
            # idata = _get_idata(self.dataset_name)
            # _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
            # trsf = transforms.Compose([*_train_trsf, *_common_trsf])      
            # combined_dataset = DummyDataset(combined_data, combined_label, trsf, use_path=False)
        else:
            combined_data = np.concatenate((cur_data, pre_data),axis=0)
            combined_label = np.concatenate((cur_labels, pre_labels),axis=0)
            # combined_data, combined_label = np.vstack((cur_dataset.images, pre_dataset.images)), np.vstack((cur_dataset.labels, pre_dataset.labels))
            # combined_label = np.concatenate(combined_label)
        idata = _get_idata(self.dataset_name)
        _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
        trsf = transforms.Compose([*_train_trsf, *_common_trsf])      
        combined_dataset = DummyDataset(combined_data, combined_label, trsf, use_path=False)

        return combined_dataset


    def _extract_vectors(self, loader):
        """使用模型提取表征
        Args:
            loader (DataLoader): Train_Loader / Test_Loader

        Returns:
            tuple (ndarray, ndarray): 表征和类别的ndarray
        """
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.cuda())
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.cuda())
                )
            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        print("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )   # empty list
        self._class_means = np.zeros((self._total_classes, self.feature_dim)) # shape, (20, 64)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        # for each old class, xx
        for class_idx in range(self._known_classes):  # 0 for the first task
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        print("Constructing exemplars...({} per classes)".format(m))
        # for current task
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )   # return dataset for one class, 500 samples
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)  # get feature maps
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)  # (100, 32, 32, 3)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        """
            herding 方式为新类构建Buffer(求类别均值)
        Args:
            data_manager (_type_): _description_
            m (_type_): 每个类别存储的样本数量
        """
        print(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取样本表征，计算类别均值
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # 按照 KNN 选择 K 个 样本 加入到selected_exemlars
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means


    def show_dataset(self, train_dataset):
        print('------task {} show dataset(random sample)-------'.format(self._cur_task))
        indices = np.random.permutation(train_dataset.images.shape[0]).tolist()[0:100]
        indices.sort()
        train_img = np.take(train_dataset.images, indices, axis= 0)
        train_img = rearrange(train_img, '(a d) h w c -> (a h) (d w) c', a = 10 , d = 10) 
        train_img = wandb.Image(train_img)
        wandb.log({"task_{}_dataset".format(self._cur_task): train_img})

    def show_client_distribution(self, idx_map, labels, classes):
        """展示每个Client上的类别分布   
        Args:
            idx_map (dict): 客户端ID与data idx的映射
            labels (ndarray): 标签列表(idx) 
            classes (list):  标签名称列表
        """
        print('---------show client distribution------------')
        start_label = min(labels)
        end_label = max(labels)
        n_clients = len(idx_map)
        # 独立同分布
        client_idcs = [[] for i in range(n_clients)]
        for client in idx_map:
            for idx in idx_map[client]:
                client_idcs[client].append(idx)

        plt.figure(figsize=(12, 8))
        plt.hist([labels[idc] for idc in client_idcs], stacked=True,
                bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
                label=["Client {}".format(i) for i in range(n_clients)],
                rwidth=0.5)
        plt.xticks(np.arange(start_label,end_label + 1), classes)
        plt.xlabel("Label type")
        plt.ylabel("Number of samples")
        plt.legend(loc="upper right")
        plt.title("Display Label Distribution on Different Clients")
        # plt.show()
        wandb.log({"task_{}_distribution_beta{}".format(self._cur_task,self.args["beta"]): wandb.Plotly(plt.gcf())})
