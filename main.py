import argparse
import wandb, os
# Pytorch 性能调优工具
from torch.autograd.profiler import profile
from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
from methods.myfcl import MyFCL
import warnings
warnings.filterwarnings('ignore')


def get_learner(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "target":
        return TARGET(args)
    elif name == "ours":
        return MyFCL(args)
    else:
        assert 0
        

def train(args):
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(     
        args["dataset"],
        True,
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    learner = get_learner(args["method"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    
    # train for each task
    for task in range(data_manager.nb_tasks):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network), 
            count_parameters(learner._network, True))) 
        learner.incremental_train(data_manager) # train for one task
        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()
        # 增加检查点
        learner.save_checkpoint("checkpoint_{}_{}".format(args["exp_name"],task))
        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))
        # !
        # break
    

def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    # todo
    parser.add_argument('--exp_name', type=str, default='test', help='name of this experiment')
    # todo
    parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb')
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="TARGET", help='wandb project')
    parser.add_argument('--group', type=str, default="exp1", help='wandb group')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar10", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    # todo
    parser.add_argument('--method', type=str, default="finetune", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet32", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--nums', type=int, default=8000, help='the num of synthetic data')
    parser.add_argument('--kd', type=int, default=25, help='for kd loss')
    parser.add_argument('--memory_size', type=int, default=300, help='the num of real data per task')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = args_parser()
    if args.dataset=="tiny_imagenet":
        args.num_class = 200 
    elif args.dataset=="cifar10":
        args.num_class = 10 
    else:
        args.num_class = 100 
    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"
    if args.method == "ours":
        dir = "run"
        if not os.path.exists(dir):
            os.makedirs(dir) 
        args.save_dir = os.path.join(dir, args.group+"_"+args.exp_name)
    
    if args.wandb == 1:
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
    args = vars(args)

    train(args)

    # 性能分析
    # with profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    #     train(args)
    # print(prof.table())
    # prof.export_chrome_trace('./resnet_profile.json')
