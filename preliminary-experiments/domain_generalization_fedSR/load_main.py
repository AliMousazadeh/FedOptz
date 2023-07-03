# to choose random subset of clients for domain generalization
import random

from torch.utils.data import ConcatDataset
import csv
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
from centralized import Centralized




def load_compressed_objects(filename):
    with gzip.open(filename, 'rb') as file:
        objects = pickle.load(file)
    return objects


def prepare_clients(clients, args):
    # I have added this code to rotate a subset of clients
    # is that ok to name the output and input of the function the same (same reference)
    for i, client in enumerate(clients):
        client.create_loaders()
        client.set_args(args)


def rotate_train_clients_subset(clients):
    # next commented line is for test
    # clients[0].apply_additional_transformation(15)

    chosen_clients_num = float(len(clients)) * (1000.0/3500)
    chosen_clients_for_rotation = random.sample(clients, int(chosen_clients_num))
    without_rotation_clients = [x for x in clients if x not in chosen_clients_for_rotation]
    rotation_degree = 0
    for i in range(int(chosen_clients_num/6), int(chosen_clients_num)):

        if i % (int(chosen_clients_num/6)) == 0:
            rotation_degree += 15
        chosen_clients_for_rotation[i].apply_additional_transformation(rotation_degree)

    clients_with_rotated_subset = chosen_clients_for_rotation + without_rotation_clients
    random.shuffle(clients_with_rotated_subset)
    # is the following line necessary and optimized?
    return clients_with_rotated_subset

def leave_one_domain_out(clients,test_domain_num):
    # chosen_clients_num = float(len(clients)) * (1000.0/3500)
    # chosen_clients_for_rotation = random.sample(clients, int(chosen_clients_num))
    domain_length = int(len(clients) / 6)
    new_train_clients = []
    new_test_clients = []
    rotation_degree = -15

    for i in range(int(len(clients))):

        if i % domain_length == 0:
            rotation_degree += 15

        if (test_domain_num - 1) * domain_length <= i < test_domain_num * domain_length:
            clients[i].apply_additional_transformation(rotation_degree)
            new_test_clients.append(clients[i])
        else:
            clients[i].apply_additional_transformation(rotation_degree)
            new_train_clients.append(clients[i])


    return new_train_clients, new_test_clients

def extract_data(train_clients, test_clients):
    train_data = []
    test_data = []

    for client in train_clients:
        train_data.append(client.get_datasets())

    for client in test_clients:
        test_data.append(client.get_datasets())

    with open('F://projectData//training_data.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(train_data)

    with open('F://projectData//test_data.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(test_data)

def extract_train_centralized(train_clients, test_clients, bs):
    train_set = [client.dataset for client in train_clients]
    test_set = [client.dataset for client in test_clients]

    train_set = ConcatDataset(train_set)
    test_set = ConcatDataset(test_set)

    print("Creating training and test loader for centralized training...")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle=False)

    print("initializing centralized training you can change parameters here...")
    learning_rate = 0.01
    weight_decay = 0.000001
    momentum = 0.9
    epochs = 20
    centralized = Centralized(train_loader, test_loader, learning_rate, weight_decay, momentum, epochs)

    print("Start Centralized Training...")
    centralized.main_centralized()
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
        return CNNModel()
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
    set_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)
    print('Done.')

    metrics = set_metrics(args)

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


    print("Preparing clients...")
    prepare_clients(train_clients, args)
    prepare_clients(test_clients, args)

    for counter in range(6):
        print("Part " + str(counter))
        test_domain_num = counter
        train_clients_temp, test_clients_temp = leave_one_domain_out(train_clients, test_domain_num)

        print("setting fedSR parameters in run_epooch_sr")
        for c in train_clients_temp:
            c.set_sigma_mu_C()
        for c in test_clients_temp:
            c.set_sigma_mu_C()

        print("Training...")
        server = Server(args, train_clients_temp, test_clients_temp, model, metrics)
        server.train()
        print("Done.")


if __name__ == '__main__':
    main()