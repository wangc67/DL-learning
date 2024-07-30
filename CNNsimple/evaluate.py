import os
from numpy import asarray, sqrt
import torch
import models
from data_loader import GetCIFAR100, GetDataLoader
from PIL.Image import open as Open
import argparse

def accuracy(predicted, labels):
    _, predict_id = torch.max(predicted, dim=1)
    return torch.tensor([torch.sum(predict_id == labels).item(), len(predict_id)])

def evaluate(model, valid_dl, device):
    prediction = torch.tensor([0, 0], dtype=int) # [correct, all]
    model.eval()
    model.to(device)
    with torch.no_grad():
        for X, Y in valid_dl:
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            prediction += accuracy(Yhat, Y) # 
    return prediction[0] / prediction[1]

def test(args):
    device = 'cuda:0' if (torch.cuda.is_available() and args.mode == 'gpu') else 'cpu'
    if args.weight is not None:
        model = torch.load(args.weight)
    else:
        model = args.net()
        model.load_state_dict(torch.load(args.weight_dict))
    TestLoader = args.get_data_function(args.batch_size, mode='test')
    Acc = evaluate(model, TestLoader, device)
    return Acc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='gpu', help='cpu or gpu')
    ap.add_argument('--get_data_function', default=GetDataLoader)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--net', default=models.resnet18)
    ap.add_argument('--weight', type=str, default='../weights/plant_best.pt')
    ap.add_argument('--weight_dict', type=str, default=None)
    args = ap.parse_args()

    acc = test(args)
    print(f'test acc {acc:.3f}')