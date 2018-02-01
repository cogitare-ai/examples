import argparse
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pprint
import cogitare
from cogitare.models.classic import LogisticRegression, FeedForward
from cogitare.monitor import Monitor
from cogitare.data import DataSet
from sklearn.datasets import fetch_mldata


# Configurations

parser = argparse.ArgumentParser(description='Cogitare PyTorch MNIST Model')

parser.add_argument('--batch-size', help='Size of the training batch', type=int, default=64, metavar='N')
parser.add_argument('--cuda', help='enable cuda', action='store_true')
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.1, metavar='N')
parser.add_argument('--model', help='select the model to train', choices=['logistic', 'mlp'], default='mlp')
parser.add_argument('--momentum', help='momentum', type=float, default=0.0, metavar='N')
parser.add_argument('--seed', help='random generator seed', type=int, default=123, metavar='N')

args = parser.parse_args()
print('Running configurations: \n')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
print('\n')
cogitare.seed(args.seed)

# Data
mnist = fetch_mldata('MNIST original')
mnist.data = (mnist.data / 255).astype(np.float32)
data = DataSet([mnist.data, mnist.target.astype(int)], batch_size=args.batch_size)
data_train, data_validation = data.split(0.8)

# Model
if args.model == 'logistic':
    model = LogisticRegression(input_size=784, num_classes=10, use_cuda=args.cuda)
elif args.model == 'mlp':
    model = FeedForward(784, 2, [75, 10], [nn.ReLU(), nn.LogSoftmax()])

model.register_default_plugins()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

monitor = Monitor('MNIST - Logistic Regression', 'Simple example of digit recognition with Logistic Regression')

model.learn(data_train, optimizer, data_validation)

print('Model trainined! Press any key to exit ...')
input()
