import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                     kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, T, C, H, W)   [Batch, Time steps (7), Channels (12), Height, Width]
        :return:
        '''
        outputs = []
        B, T, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)

        for t in range(T):
            combined = torch.cat([inputs[:, t], hx], dim=1)  # Concatenate input with hidden state
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()  # (B, T, C, H, W)


class ConvLSTMWildfirePredictor(nn.Module):
    def __init__(self, num_bands=12, hidden_dim=64, num_classes=1):
        super().__init__()
        self.convlstm = ConvLSTMBlock(in_channels=num_bands, num_features=hidden_dim)
        self.conv1 = nn.Conv2d(hidden_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        '''
        :param x: (B, 7, 12, 128, 128)  # 7-day sequence of Sentinel-2 images
        :return: Probability of wildfire (0 or 1)
        '''
        x = self.convlstm(x)[:, -1]  # Get the last time step output
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

####https://github.com/czifan/ConvLSTM.pytorch
