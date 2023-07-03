import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
import torch.nn as nn

from models.cnn_model import CNNModel

class Centralized:

    def __init__(self,  train_loader, test_loader, learning_rate, weight_decay, momentum, epochs):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs

    def get_loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function

    def get_optimizer(self, net, lr, wd, momentum):
        optimizer = torch.optim.SGD(net.parameters(), lr = lr,
                              weight_decay=wd, momentum=momentum)
        return optimizer

    def train(self, net, data_loader, optimizer, loss_function, device = 'cuda:0'):
        samples = 0
        cumulative_loss = 0
        cumulative_accuracy = 0
        net.train()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
            outputs = net(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()#computes the gradients
            optimizer.step()#updates the weights of the neural network model
            optimizer.zero_grad()#removing the old gradeinets
            samples += inputs.shape[0]#total number of data seen in the current batch
            cumulative_loss += loss.item()#current online running loss of the current epoch
            _, predicted = outputs.max(1)#predicted classes
            cumulative_accuracy += predicted.eq(targets).sum().item()# cumulative acurracy by adding correct predictions
        return cumulative_loss/samples, cumulative_accuracy/samples * 100


    def test(self, net, data_loader, loss_function, device='cuda:0'):
        samples = 0
        cumulative_loss = 0
        cumulative_accuracy = 0

        net.eval()

        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(device)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device)
                targets = targets.reshape(images.shape[0])
                samples += images.shape[0]
                outputs = net(images)
                loss = loss_function(outputs, targets)
                cumulative_loss += loss.item()

                _, predicted = outputs.max(1)
                cumulative_accuracy += predicted.eq(targets).sum().item()

        return cumulative_loss / samples, cumulative_accuracy / samples * 100


    def main_centralized(self, device='cuda:0'):
        net = CNNModel().to(device)  # ??

        optimizer = self.get_optimizer(net, self.learning_rate, self.weight_decay, self.momentum)
        loss_function = self.get_loss_function()

        for e in range(self.epochs):
            train_loss, train_accuracy = self.train(net, self.train_loader, optimizer, loss_function)
            # val_loss, val_accuracy = self.test(net, loss_function)
            print('Epoch: {:d}'.format(e + 1))
            print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
                                                                         train_accuracy))
            # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
            #                                                                  val_accuracy))
            print('-----------------------------------------------------')

        print('After training:')
        train_loss, train_accuracy = self.test(net, self.train_loader, loss_function)
        # val_loss, val_accuracy = test(net, val_loader, loss_function)
        test_loss, test_accuracy = self.test(net, self.test_loader, loss_function)

        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
                                                                     train_accuracy))
        # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
        #                                                                  val_accuracy))
        print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
        print('-----------------------------------------------------')