import copy, wandb
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
from utils.compressor import get_compressor_model


def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    return sorted(temp.items(),key=lambda x:x[0])



class MyFCL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.acc = []
        self._compressor = get_compressor_model(args['dataset'], False)
    # 压缩
    def compress_data(self, replay_samples):
        encoding = self._compressor(replay_samples)
        quantized_inputs, _ = self.vq_layer(encoding)
        
        return quantized_inputs

    # 解压缩
    def decompress_data(self, quantized_inputs):
        return self._compressor.decode(quantized_inputs)
    #

    # 填充缓冲区
    def build_memory(self, selected_exemplar_data, selected_exemplar_label):
        
        self._targets_memory

    # 获得缓冲区中的数据
    def get_memory(self, ):
        return 
    
    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()

    # 新的FCL损失
    def _PCR_Loss():
        pass

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
            mode="train",
            # appendent=self._get_memory()
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
        self._fl_train(train_dataset, data_manager, self.test_loader)


    def _local_update(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += images.shape[0]
            if iter ==0 and com_id==0 : print("task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task ,client_id, total, tmp))
        return model.state_dict()

    # pdb.set_trace()
    
    
    def print_data_stats(client_id, train_data_loader):
        def sum_dict(a,b):
            
            temp = dict()
            for key in a.keys() | b.keys():
                temp[key] = sum([d.get(key, 0) for d in (a, b)])
            return temp
        temp = dict()
        for batch_idx, (_, images, labels) in enumerate(train_data_loader):
            unq, unq_cnt = np.unique(labels, return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            temp = sum_dict(tmp, temp)
        return sorted(temp.items(),key=lambda x:x[0])




    def _local_finetune(self, model, train_data_loader, client_id, tmp, com_id):
        pass
    
    
    
    def _fl_train(self, train_dataset, data_manager,test_loader):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                # update local train data
                if self._cur_task == 0:
                    local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                else:

                    current_local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                    previous_local_dataset = self.get_all_previous_dataset(data_manager, idx) 

                    local_dataset = self.combine_dataset(previous_local_dataset, current_local_dataset, self.memory_size)
                    local_dataset = DatasetSplit(local_dataset, range(local_dataset.labels.shape[0]))

                local_train_loader = DataLoader(local_dataset, batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                tmp = print_data_stats(idx, local_train_loader)
                if com !=0:
                    tmp = ""
                if self._cur_task == 0:                    
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                local_weights.append(copy.deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})