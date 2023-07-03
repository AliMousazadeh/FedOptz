import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
import kornia
import random
import torchvision
from torchvision import transforms

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, test_client=False):
        ####
        self.device = "cuda"
        ####


        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.test_client = test_client

        self.train_loader = None
        self.test_loader = None
        self.train_loader_grad = None

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()


    def __str__(self):
        return self.name


    def create_loaders(self):
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not self.test_client else None
        self.train_loader_grad = DataLoader(self.dataset, batch_size=32, shuffle=False, drop_last=False) \
            if not self.test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=32, shuffle=False, drop_last=False)


    def set_args(self, args):
        self.args = args

    def get_num_batches(self):
        return len(self.train_loader)


    def grad_seq(self, model):

        data_index = np.random.choice(len(self.dataset), self.args.bs, replace=False)

        images_list, labels_list = [], []
        for i in range(len(data_index)):
            image, label = self.dataset[data_index[i]]
            images_list.append(image.unsqueeze(dim=0))
            labels_list.append(label)

        images = torch.cat(images_list)
        labels = torch.cat(labels_list)

        images = images.to(self.device)
        labels = labels.type(torch.LongTensor).to(self.device)

        out = model(images)

        loss = self.criterion(out, labels) / self.args.clients_per_round
        loss.backward()


    def grad_rev(self, model, client_num):
        model.to(self.device)

        lambda_val = 1
        for cur_step, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device).squeeze()

            out, pred_client = model(images)
            loss_client = self.criterion(pred_client, torch.zeros_like(labels) + client_num)
            loss_client = lambda_val * loss_client / (len(self.train_loader) * self.args.clients_per_round)


            loss_client.backward()


    def calculate_loss(self, model):
        model.to(self.device)
        total_loss = 0

        for cur_step, (images, labels) in enumerate(self.train_loader_grad):

            images = images.to(self.device)

            labels = labels.type(torch.LongTensor)
            labels = labels.to(self.device)
            labels = labels.reshape(images.shape[0])

            out = model(images)

            loss = self.criterion(out, labels) / len(self.train_loader_grad)
            total_loss += loss.item()

        return total_loss

    def calculate_gradient(self, model):
        model.to(self.device)
        total_loss = 0
        flag = False

        for cur_step, (images, labels) in enumerate(self.train_loader_grad):

            images = images.to(self.device)
            labels = labels.type(torch.LongTensor)
            # labels = labels.to(self.device).squeeze()
            labels = labels.to(self.device)
            labels = labels.reshape(images.shape[0])

            out = model(images)

            loss = self.criterion(out, labels) / len(self.train_loader_grad)
            total_loss += loss.item()

            loss.backward()

            flag = True

        if not flag:
            raise NotImplementedError

        return model, total_loss


    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)


    def _get_outputs(self, images, model):
        if self.args.model == 'cnn':
            return model(images)
        if self.args.model == 'deeplabv3_mobilenetv2':
            return model(images)['out']
        if self.args.model == 'resnet18':
            return model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer, model, client_num):

        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :param model: training model for this client
        :param client_num: number of the current client
        """
        total_loss = 0
        cur_step = 1
        for cur_step, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(self.device).squeeze()

            out = model(images)
            loss = self.criterion(out, labels)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model, total_loss/(cur_step + 1)


    def train(self, model, client_num):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.m)
        model.to(self.device)

        total_loss = 0
        model.train()
        for epoch in range(self.args.num_epochs):
            model, total_loss = self.run_epoch(epoch, optimizer, model, client_num)

        return model, total_loss


    def test(self, metric, model):
        """
        This method tests the model on the local dataset of the client.
        param metric: StreamMetric object
        """
        model.eval()
        model.to(self.device)

        with torch.inference_mode():

            for i, (images, labels) in enumerate(self.test_loader):

                outputs = model(images.to(self.device))
                self.update_metric(metric, outputs, labels)