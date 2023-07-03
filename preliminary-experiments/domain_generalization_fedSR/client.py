import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction

import torch.nn.functional as F

class Client:

    def __init__(self, args, dataset, test_client=False):
        ####
        self.device = "mps"
        ####


        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.test_client = test_client

        self.train_loader = None
        self.test_loader = None

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        self.r_mu = nn.Parameter(torch.zeros(62, 1024))
        self.r_sigma = nn.Parameter(torch.ones(62, 1024))
        self.C = nn.Parameter(torch.ones([]))

    def __str__(self):
        return self.name


    def create_loaders(self):
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not self.test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)


    def set_args(self, args):
        self.args = args


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

    def run_epoch(self, cur_epoch, optimizer, model):
        self.device = "mps"
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :param model: training model for this client
        """
        total_loss = 0
        cur_step = 1
        for cur_step, (images, labels) in enumerate(self.train_loader):
            # TODO: missing code here!

            images = images.to(self.device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(self.device).squeeze()

            out = model(images)

            # print(next(model.parameters()).device)

            loss = self.criterion(out, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model, total_loss/(cur_step + 1)

    def run_epoch_sr(self, cur_epoch, optimizer, model):
        self.device = "mps"
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :param model: training model for this client
        """
        total_loss = 0
        cur_step = 1
        for cur_step, (images, labels) in enumerate(self.train_loader):
            # TODO: missing code here!

            images = images.to(self.device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(self.device).squeeze()

            z, z_mu, z_sigma = model.featurize(images)
            out = model(images)

            # print(next(model.parameters()).device)

            loss = self.criterion(out, labels)
            regL2R = torch.zeros_like(loss)
            regCMI = torch.zeros_like(loss)

            regL2R = z.norm(dim=1).mean()
            loss = loss + self.args.l2r * regL2R

            r_sigma_softplus = F.softplus(self.r_sigma)
            r_mu = self.r_mu[labels.cpu()]
            r_mu = r_mu.to("mps")
            r_sigma = r_sigma_softplus[labels.cpu()]
            r_sigma = r_sigma.to("mps")
            z_mu_scaled = z_mu * self.C
            z_sigma_scaled = z_sigma * self.C
            regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
            regCMI = regCMI.sum(1).mean()
            loss = loss + self.args.cmi*regCMI

            total_loss += loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        return model, total_loss/(cur_step + 1)

    def train(self, model):
        self.device = "mps"
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        # TODO: missing code here!
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.m)
        model.to(self.device)
        # optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)

        total_loss = 0
        model.train()
        for epoch in range(self.args.num_epochs):
            # TODO: missing code here!
            model, total_loss = self.run_epoch_sr(epoch, optimizer, model)

        return model, total_loss


    def test(self, metric, model):
        self.device = "mps"
        """
        This method tests the model on the local dataset of the client.
        param metric: StreamMetric object
        """
        # TODO: missing code here!
        model.eval()
        model.to(self.device)

        with torch.inference_mode():
            for i, (images, labels) in enumerate(self.test_loader):
                # TODO: missing code here!
                outputs = model(images.to(self.device))
                self.update_metric(metric, outputs, labels[0])

    def apply_additional_transformation(self, rotation_degree):
        self.dataset.apply_new_transformation(rotation_degree)

# This function is for extracting date for centralized training
    def get_datasets(self):
        return self.dataset
#.get_samples()
    def set_sigma_mu_C(self):
      self.r_sigma = nn.Parameter(torch.ones(62, 1024))
      self.r_mu = nn.Parameter(torch.zeros(62, 1024))
      self.C = nn.Parameter(torch.ones([]))
