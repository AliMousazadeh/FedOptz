import copy
from collections import OrderedDict

import numpy as np
from models.cnn_model import CNNModel
import torch
from tqdm import tqdm


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.metrics = metrics

        self.global_model = model
        self.updating_model_dict = copy.deepcopy(self.global_model.state_dict())


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
        with tqdm(clients, unit="client") as tepoch:
            for i, client in enumerate(tepoch):
                trained_model, client_loss = client.train(copy.deepcopy(self.global_model))
                total_loss += client_loss

                num_clients = min(self.args.clients_per_round, len(self.train_clients))
                new_model_dict = trained_model.state_dict()

                if i == 0:
                    for k in self.updating_model_dict:
                        self.updating_model_dict[k] = new_model_dict[k] / num_clients
                else:
                    for k in self.updating_model_dict:
                        self.updating_model_dict[k] += new_model_dict[k] / num_clients

                tepoch.set_description(f"Round {r+1}/{self.args.num_rounds} -> train client {i+1}")
                tepoch.set_postfix(loss=total_loss / (i+1))


        return total_loss

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        # TODO: missing code here!
        raise NotImplementedError

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        for r in (range(self.args.num_rounds)):
            clients = self.select_clients()
            total_loss = self.train_round(clients, r)

            model = CNNModel()
            model.load_state_dict(self.updating_model_dict)

            ## is a deep copy necessary?
            self.global_model = copy.deepcopy(model)


            if (r+1)%self.args.test_interval == 0 and r+1 > 500:
                self.test()






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


    def test(self):
        """
            This method handles the test on the test clients
        """
        self.metrics['test'].reset()

        with tqdm(self.test_clients, unit="client") as tepoch:
            for i, client in enumerate(tepoch):
                client.test(self.metrics['test'], self.global_model)
                self.metrics['test'].get_results()

                tepoch.set_description(f"test client {i + 1}")
                tepoch.set_postfix(accuracy=self.metrics['test'].results["Overall Acc"])

        self.metrics['test'].reset()
