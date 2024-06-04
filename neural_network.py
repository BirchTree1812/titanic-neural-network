import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# a fairly basic neural network that uses MSELoss and Adam optimizer. 
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        # declare linear layers, batch normalization and dropout. 
        # Linear starts with high size, then decreases. Batch norm layers follow linear layers. 
        # if I increase the starting linear value and then add more layers, I could improve precision of the model
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc7 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # relu->batch normalization -> linear
        x = F.relu(self.bn1(self.fc1(x)))
        # dropout removes some neurons to mitigate overfitting
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        # straightens out the values, so that they're between 1 and 0.
        x = torch.sigmoid(self.fc7(x))
        return x

