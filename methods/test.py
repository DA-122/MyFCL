import copy, wandb
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed

from sklearn.metrics import confusion_matrix
from torchsummary import summary




class TestResult(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._checkpoint_prex = args["checkpoint"]

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

        # # wandb展示数据集
        # if self.wandb == 1:
        #     self.show_dataset(train_dataset)

        # 获得新任务的测试数据集
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        ) 

        # 模型载入checkpoint
        # file_name = "checkpoint_0_finetune_finetune_10clients_5tasks_beta0.5_{}_{}.pkl".format(self._cur_task,self._cur_task)
        file_name = "./checkpoint/"+self._checkpoint_prex+"_{}_{}.pkl".format(self._cur_task,self._cur_task)
        ckp_dict = torch.load(file_name)
        self._network.load_state_dict(ckp_dict["model_state_dict"])
        
        # TSNE可视化
        self.show_Tsne(test_dataset)









