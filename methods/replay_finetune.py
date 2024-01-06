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
# finetune 微调方法 fedavg

# init_epoch = 200
# com_round = 100  
# num_users = 5 # 5, 
# frac = 1 # 

# local_bs = 128  # cifar100, 5w, 5 tasks, 1w for each task, 2k for each client
# local_ep = 5
# batch_size = 128
# num_workers = 4

tau=1

def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        # | 并集
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    print(sorted(temp.items(),key=lambda x:x[0]))


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class ReplayFinetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.memory_size = args["memory_size"]
        self.acc = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()


    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""
        KLDiv = nn.KLDivLoss(reduction="batchmean")
        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self._total_classes)
        pred_probs = F.log_softmax(logits / tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self._total_classes)
            dg_probs = torch.softmax(dg_logits / tau, dim=1)

        loss = (tau ** 2) * KLDiv(pred_probs, dg_probs)

        return loss

    def get_all_previous_dataset(self, data_manager, idx):
        # for second task, self._cur_task=1
        bgn_cls, end_cls = 0, self.each_task
        train_dataset = data_manager.get_dataset(
            np.arange(bgn_cls, end_cls),
            source="train",
            mode="train",
        )
        setup_seed(self.seed)
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        all_previous_dataset = DatasetSplit(train_dataset, user_groups[idx])
        # for third task
        for i in range(2, self._cur_task+1):  # 2-4
            setup_seed(self.seed)
            bgn_cls += self.each_task  # 20-40
            end_cls += self.each_task
            train_dataset_next = data_manager.get_dataset(
                np.arange(bgn_cls, end_cls),
                source="train",
                mode="train",
            )
            user_groups_next = partition_data(train_dataset_next.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
            tmp_dataset = DatasetSplit(train_dataset_next, user_groups_next[idx])  # <utils.data_manager.DummyDataset>
            all_previous_dataset = self.combine_dataset(all_previous_dataset, tmp_dataset, 0) # combine two datasets
            all_previous_dataset = DatasetSplit(all_previous_dataset, range(all_previous_dataset.labels.shape[0]))
            # 2417->   all_previous_dataset.idxs[0:4]= [9013, 7479, 5185, 7241]
        return all_previous_dataset

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

        # 获得新任务的测试数据集
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        ) 

        # 模型载入checkpoint
        # if self._cur_task < 4:
        #     ckp_dict = torch.load("checkpoint_0_finetune_finetune_10clients_5tasks_beta0.5_{}_{}.pkl".format(self._cur_task,self._cur_task))
        #     self._network.load_state_dict(ckp_dict["model_state_dict"])
        
        # TSNE可视化
        # self.show_Tsne(test_dataset)

        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=0
        )
        setup_seed(self.seed)
        #!
        self._fl_train(train_dataset, self.test_loader, data_manager)
        

        # if self._cur_task == 0:
        #     # self._fl_train(train_dataset, self.test_loader)
        #     # torch.save(self._network.state_dict(), 'finetune.pkl')
        #     # print("save checkpoint >>>")

        #     self._network.cuda()
        #     state_dict = torch.load('finetune.pkl')
        #     self._network.load_state_dict(state_dict)
        #     test_acc = self._compute_accuracy(self._network, self.test_loader)
        #     print("For task 1, loading ckpt, acc:{}".format(test_acc))

        #     # return 
        # else:
        #     # return 
        #     acc = self._compute_accuracy(self._old_network, self.pre_loader)
        #     print("loading ckpt, acc:{}".format(acc))
            
        #     self._fl_train(train_dataset, self.test_loader)

        

    # def _local_update(self, model, train_data_loader):
    #     model.train()
    #     cp_model =  copy.deepcopy(model)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #     for iter in range(local_ep):
    #         for batch_idx, (_, images, labels) in enumerate(train_data_loader):
    #             images, labels = images.cuda(), labels.cuda()
    #             output = model(images)["logits"]
    #             loss_ce = F.cross_entropy(output, labels)
    #             with torch.no_grad():
    #                 dg_logits = cp_model(images.detach())["logits"]
    #             # only learn from out-distribution knowledge, overcome local forgetting
    #             loss_ntd = self._ntd_loss(output, dg_logits, labels)
    #             loss = loss_ce + loss_ntd 
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     return model.state_dict()

    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()


    def per_cls_acc(self, val_loader, model):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i, (_, input, target) in enumerate(val_loader):
                input, target = input.cuda(), target.cuda()
                # compute output
                output = model(input)["logits"]
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)

        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)

        cls_acc = cls_hit / cls_cnt
        return cls_acc
        # pdb.set_trace()
        # out_cls_acc = 'Per Class Accuracy: %s' % ((np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        # print(out_cls_acc)
        

    def _local_finetune(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # print_data_stats(0, train_data_loader)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss = F.cross_entropy(output[:, self._known_classes :], fake_targets.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # self.per_cls_acc(self.test_loader, model)
        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader,data_manager):
        # if self._cur_task == 0:
        #     return

        self._network.cuda()
        cls_acc_list = []
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        # wandb中显示client_distribution
        self.show_client_distribution(user_groups,train_dataset.labels,train_dataset.classes)

        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                # todo 并行跑代码，充分利用显存
                if self._cur_task == 0:
                    local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                else:

                    current_local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                    previous_local_dataset = self.get_all_previous_dataset(data_manager, idx) 

                    local_dataset = self.combine_dataset(previous_local_dataset, current_local_dataset, self._memory_per_class*self._known_classes)
                    local_dataset = self.combine_dataset(previous_local_dataset, current_local_dataset, self._memory_per_class*self._known_classes)
                    local_dataset = DatasetSplit(local_dataset, range(local_dataset.labels.shape[0]))

                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=0)
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader)
                local_weights.append(copy.deepcopy(w))
                # del local_train_loader
                # torch.cuda.empty_cache()

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                # 每个类别的准确率
                cls_acc = self.per_cls_acc(self.test_loader, self._network)
                cls_acc_list.append(cls_acc)

                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})

        acc_arr = np.array(cls_acc_list)
        acc_max = acc_arr.max(axis=0)
        if self._cur_task == 4:
            acc_max = self.per_cls_acc(self.test_loader, self._network)
        print("For task: {}, acc list max: {}".format(self._cur_task, acc_max))
        self.acc.append(acc_max)



