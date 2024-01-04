import copy, wandb
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed

class MyFCL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.acc = []
    
    # 新的FCL损失
    def _PCR_Loss():
        pass

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        # summary(self._network, (3,32,32), device='cpu')
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 获得新任务的训练数据集
        train_dataset = data_manager.get_dataset(   #* get the data for one task
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
        )
        # wandb展示数据集
        if self.wandb == 1:
            self.show_dataset(train_dataset)

        # # 获得新任务的测试数据集
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)

    def _fl_train():
        pass