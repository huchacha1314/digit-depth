#3层 MLP 将RGB-> xyz法向量
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    dropout_p = 0.05

    def __init__(
            self, input_size=5, output_size=3, hidden_size=32):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)#5RGBXY（XY在图像中的位置）->32 
        self.fc2 = nn.Linear(hidden_size, hidden_size)#32->32
        self.fc3 = nn.Linear(hidden_size, hidden_size)#32->32
        self.fc4 = nn.Linear(hidden_size, output_size)#32->3 N_x N_y N_z的法向量
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x
