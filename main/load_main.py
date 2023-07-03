import pickle
import gzip
import time
import torch
import random
import numpy as np
from torchvision.models import resnet18
from torch import nn
from server import Server
from utils.args import get_parser
from models.deeplabv3 import deeplabv3_mobilenetv2
from models.cnn_model import CNNModel
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm
from client import Client

import requests



def load_compressed_objects(filename):
    with gzip.open(filename, 'rb') as file:
        objects = pickle.load(file)
    return objects


def prepare_clients(clients, args):
    for i, client in enumerate(clients):
        client.set_args(args)
        client.create_loaders()


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        return CNNModel(args)
    raise NotImplementedError


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def main():


    parser = get_parser()
    args = parser.parse_args()


    portion_str = '_' + args.data_portion if args.data_portion else '_'

    if args.niid:
        path_train = "pickles/pickles" + portion_str + "/niid/train_clients.pkl.gz"
        path_test = "pickles/pickles" + portion_str + "/niid/test_clients.pkl.gz"
    else:
        path_train = "pickles/pickles" + portion_str + "/iid/train_clients.pkl.gz"
        path_test = "pickles/pickles" + portion_str + "/iid/test_clients.pkl.gz"


    print("Loading data...")
    start_time = time.time()
    train_clients = load_compressed_objects(path_train)
    test_clients = load_compressed_objects(path_test)
    elapsed_time = time.time() - start_time
    print(f"Done. Elapsed time: {elapsed_time/60:.2f}m")


    set_seed(args.seed)


    metrics = set_metrics(args)

    print("Preparing clients...")
    prepare_clients(train_clients, args)
    prepare_clients(test_clients, args)
    print("Done.")

    print(f'Initializing model...')
    model = model_init(args)
    print('Done.')

    print("Training...")
    print(args.method)
    server = Server(args, train_clients, test_clients, model, metrics)
    server.alpha = args.alpha

    server.train()
    print('Test accuracy:')
    print(f'Mean: {torch.mean(server.test_accuracies):.4}\tstd: {torch.std(server.test_accuracies, unbiased=True):.4}')
    print("Done.")



if __name__ == '__main__':
    main()