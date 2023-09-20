from data import *


class MnistRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act2 = nn.Softmax(dim=-1)

    def forward(self, x):
        out1 = self.fc1(self.act1(x))
        out2 = f.dropout(out1, training=self.training)
        out = self.act2(self.fc2(out2))

        return out


def create_model(input_size, hidden_size, output_size, learning_rate=0.05):
    cache = {}

    model = MnistRecognizer(input_size, hidden_size, output_size)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
        cost = cost.cuda()

    cache['model'] = model
    cache['cost'] = cost
    cache['optim'] = optimizer

    return cache


def batch_training(model_cache, batch_cache, epochs=200, define_batch_num=0):
    model = model_cache['model']
    cost = model_cache['cost']
    optim = model_cache['optim']

    batch_x = batch_cache['batch_x']
    batch_y = batch_cache['batch_y']
    batch_num = batch_cache['batch_num']

    for i in range(define_batch_num if define_batch_num else batch_num):
        x = batch_x[i]
        y = batch_y[i]
        for j in range(epochs+1):
            out = model(x)
            loss = cost(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if not j % 20:
                print('batch ', i, ' epoch ', j, ' loss: ',
                      round(loss.item(), 4))


def testing(model, x, y):
    test_size = len(y)
    model.eval()
    with torch.no_grad():
        pred = de_one_hot(model(x))

        exact = torch.sum(torch.eq(pred, y)).item()
        exact_rate = exact / test_size

        return exact_rate


def recognizing(model, x):
    model.eval()
    with torch.no_grad():
        out = model.forward(x)
        value = torch.argmax(out).item()
    return value
