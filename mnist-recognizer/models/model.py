import torch
from torch import nn
from torch.nn import functional as f


class MnistRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MnistRecognizer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act2 = nn.Softmax(dim=-1)

    def forward(self, x):
        out1 = self.fc1(self.act1(x))
        out2 = f.dropout(out1, training=self.training)
        out = self.act2(self.fc2(out2))

        return out


class MnistRecognizerCNN(nn.Module):
    def __init__(self, input_channel, hidden_channel1, output_channel, kernel=4, padding=1, stride=1, pool_ker=2):
        super(MnistRecognizerCNN, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=hidden_channel1,
                kernel_size=kernel,
                padding=padding,
                stride=stride
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_ker)
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channel1,
                out_channels=output_channel,
                kernel_size=kernel,
                padding=padding,
                stride=stride
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_ker)
        )
        f1_out = (28 - kernel + 2 * padding) // stride
        f1_pool_out = f1_out - f1_out // 2
        f2_out = (f1_pool_out - kernel + 2 * padding) // stride
        f2_poll_out = f2_out - f2_out // 2
        linear_in = f2_poll_out ** 2 * output_channel
        self.f3 = nn.Sequential(
            nn.Linear(linear_in, 128)
        )
        self.f4 = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1], x.shape[-1])
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = out2.view(out2.size(0), -1)
        out4 = self.f3(out3)
        out5 = self.f4(out4)
        out6 = f.dropout(out5, training=self.training)
        out = f.softmax(out6, dim=-1)

        return out


# def recognizing(model, x):
#     model.eval()
#     with torch.no_grad():
#         out = model.forward(x)
#         value = torch.argmax(out).item()
#     return value
