import copy
from collections import OrderedDict

import numpy as np
from models.cnn_model import CNNModel
import torch
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.metrics = metrics

        self.global_model = model
        self.updating_model_dict = copy.deepcopy(self.global_model.state_dict())

        self.model_dicts = []
        self.model_gradients = []
        self.test_accuracies = torch.zeros(max(self.args.test_interval, 1))
        self.fed_avg_model = None

        self.global_losses = torch.zeros(self.args.clients_per_round)
        self.client_losses = torch.zeros(self.args.clients_per_round)
        self.fed_avg_losses = torch.zeros(self.args.clients_per_round)
        self.token_model_losses = torch.zeros(self.args.clients_per_round)
        self.alpha = 0.

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def inner_prod_grad(self, i, j):

        output = 0.0
        for param1, param2 in zip(self.model_gradients[i].parameters(), self.model_gradients[j].parameters()):
            output  += -torch.dot(param1.grad.view(-1), param2.grad.view(-1))

        return output

    def initialize_weights_to_zero(self, model):
        for param in model.parameters():
            param.data.fill_(0.0)

    def distance_weights(self, i, j):
        output = 0.0
        n = 0
        for param1, param2 in zip(self.model_gradients[i].parameters(), self.model_gradients[j].parameters()):
            output += torch.sum(torch.pow(param1.view(-1) - param2.view(-1), 2)).item()

        return np.sqrt(output)

    def inner_prod(self, i, j):
        if self.args.method == 'myavg':
            parameter_generator = self.model_gradients[i].named_parameters()
            model_dict = self.model_dicts[j]

            output = 0
            for name, parameter in parameter_generator:
                A = model_dict[name]
                B = parameter.grad
                D = torch.sum(A * B)

                output += D

            return output
        elif self.args.method == 'myavgalt':
            parameter_generator_i = self.model_gradients[i].named_parameters()
            model_dict_i = self.model_dicts[i]
            model_dict_j = self.model_dicts[j]

            output = 0
            for name, parameter in parameter_generator_i:
                A = model_dict_i[name]
                B = model_dict_j[name]
                C = parameter.grad
                E = torch.sum(C * (B-A))

                output += E

            return output
        else:
            raise NotImplementedError


    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)


    def train_round(self, clients, r):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        total_loss = 0
        for i, client in enumerate(clients):
            trained_model, client_loss = client.train(copy.deepcopy(self.global_model), i)
            total_loss += client_loss

            self.model_dicts.append(trained_model.state_dict())

            num_clients = min(self.args.clients_per_round, len(self.train_clients))
            new_model_dict = trained_model.state_dict()

            if i == 0:
                for k in self.updating_model_dict:
                    self.updating_model_dict[k] = new_model_dict[k] / num_clients
            else:
                for k in self.updating_model_dict:
                    self.updating_model_dict[k] += new_model_dict[k] / num_clients

        return total_loss



    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        with tqdm(range(self.args.num_rounds), unit='round', position=0, leave=True) as progress_bar:
            for r in progress_bar:
                clients = self.select_clients()
                total_loss = self.train_round(clients, r)
                total_loss /= self.args.clients_per_round

                progress_bar.set_postfix({'loss': total_loss}, refresh=True)

                fed_avg_model = CNNModel(self.args).to("cuda")
                fed_avg_model.load_state_dict(self.updating_model_dict)

                self.fed_avg_model = copy.deepcopy(fed_avg_model)
                self.model_dicts.append(self.fed_avg_model.state_dict())


                if self.args.method == 'fedavg':
                    self.global_model = copy.deepcopy(fed_avg_model)


                elif self.args.method == 'fedoptz':

                    for i, client in enumerate(clients):
                        model = CNNModel(self.args)
                        model.load_state_dict(self.model_dicts[i])
                        model_gradient, loss = client.calculate_gradient(copy.deepcopy(model))

                        self.model_gradients.append(model_gradient)
                        self.client_losses[i] = loss



                    M = torch.zeros(self.args.clients_per_round, self.args.clients_per_round)
                    for i in range(self.args.clients_per_round):
                        for j in range(self.args.clients_per_round):
                            M[i, j] = -self.inner_prod_grad(i, j)


                    D = torch.sum(M * torch.eye(self.args.clients_per_round), dim = 0)
                    C_vec = self.alpha * torch.Tensor(self.client_losses) / torch.sqrt(D)
                    C = torch.min(C_vec)
                    m_i = torch.sum(M, dim=0)
                    lamb = (1 / (2*C)) * torch.sqrt(torch.sum(m_i**2 / D))
                    x_i = m_i / (2*lamb*D)

                    my_model = CNNModel(self.args).to("cuda")
                    self.initialize_weights_to_zero(my_model)
                    my_model_dict = my_model.state_dict()

                    for i in range(self.args.clients_per_round):
                        parameter_generator = self.model_gradients[i].named_parameters()
                        for name, param in parameter_generator:
                            my_model_dict[name] += (self.model_dicts[i][name]/self.args.clients_per_round) - x_i[i]*param.grad


                    my_model.load_state_dict(my_model_dict)
                    self.global_model = copy.deepcopy(my_model)

                else:
                    raise NotImplementedError


                if self.args.test_interval == -1:
                    if (r + 1) == self.args.num_rounds:
                        self.test(r)
                else:
                    if self.args.num_rounds - (r+1) < self.args.test_interval:
                        self.test(r)



                self.model_dicts = []
                self.model_gradients = []






    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        NOTE: need to retest. might not be accurate
        """
        self.metrics['eval_train'].reset()

        with tqdm(self.train_clients, unit="client") as tepoch:
            for i, client in enumerate(tepoch):
                client.test(self.metrics['eval_train'], self.global_model)
                self.metrics['eval_train'].get_results()

                tepoch.set_description(f"train client {i + 1}")
                tepoch.set_postfix(accuracy=self.metrics['eval_train'].results["Overall Acc"])

        self.metrics['eval_train'].reset()


    def test(self, num_round):
        """
            This method handles the test on the test clients
        """
        self.metrics['test'].reset()

        with tqdm(self.test_clients, unit="client", leave = False, position=0) as tepoch:
            for i, client in enumerate(tepoch):
                client.test(self.metrics['test'], self.global_model)

                self.metrics['test'].get_results()

                tepoch.set_description(f"test client {i + 1}")
                tepoch.set_postfix(accuracy=self.metrics['test'].results["Overall Acc"])


        idx = self.args.test_interval - (self.args.num_rounds - num_round)
        self.test_accuracies[max(idx, 0)] = self.metrics['test'].results["Overall Acc"]

        self.metrics['test'].reset()

