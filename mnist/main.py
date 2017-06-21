import argparse
import torch.optim as optim
import pprint
import cogitare
from cogitare.models.linear.logistic import LogisticRegression
from cogitare.dataset import Dataset
from sklearn.datasets import fetch_mldata


# Configurations

parser = argparse.ArgumentParser(description='PyTorch MNIST Model')

parser.add_argument('--batch-size', help='Size of the training batch', type=int, default=64, metavar='N')
parser.add_argument('--cuda', help='enable cuda', action='store_true', default=False)
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.1, metavar='N')
parser.add_argument('--model', help='select the model to train', choices=['logistic', 'mlp'], default='logistic')
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
mnist.data = mnist.data / 255
data = Dataset(mnist.data, mnist.target.astype(int), batch_size=args.batch_size)

# Model
l = LogisticRegression(input_size=784, num_classes=10, cuda=args.cuda)
optimizer = optim.SGD(l.parameters(), lr=args.learning_rate, momentum=args.momentum)
l.learn(data, optimizer)
