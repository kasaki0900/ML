import data_processing
import models
import torch
from torch import nn
from torch.nn import functional as f


class Trainer:
    def __init__(self, model, cost=nn.CrossEntropyLoss(), optim=torch.optim.SGD, learning_rate=0.05):
        self.model = model
        self.cost = cost
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.cost = self.cost.cuda()
        self.optimizer = optim(model.parameters(), lr=learning_rate)

    def load_model(self, model_path):
        pass

    def train_a_batch(self, x, y, epochs):
        for i in range(epochs + 1):
            out = self.model(x)
            loss = self.cost(y, out)
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            if not i % 20:
                print('epoch', i, 'loss', round(loss.item(), 4))
                # print('epoch', i, 'loss', loss)

    def train(self, train_data_loader, epochs=200, n_batch=0):
        for idx, (data, targets) in enumerate(train_data_loader):
            if not n_batch or idx < n_batch:
                print('batch:', idx)
                data = data.cuda().float()
                data.requires_grad = True
                targets = f.one_hot(targets, num_classes=10).cuda().float()
                self.train_a_batch(data, targets, epochs)
            else:
                break

    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            x = dataset.data.cuda().float()
            y = dataset.targets.cuda()

            predict = torch.argmax(self.model(x), dim=1)
            print(predict.shape)
            accuracy = torch.sum(torch.eq(predict, y)).item() / len(dataset)

            return accuracy
