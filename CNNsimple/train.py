import time
import torch
from torch import nn
import argparse
import models
from data_loader import GetDataLoader, GetCIFAR100
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def accuracy(predicted, labels):
    _, predict_id = torch.max(predicted, dim=1)
    return torch.tensor([torch.sum(predict_id == labels).item(), len(predict_id)])

def evaluate(model, valid_dl, Loss):
    prediction = torch.tensor([0, 0], dtype=int) # [correct, all]
    val_loss = 0.
    model.eval()
    with torch.no_grad():
        for X, Y in valid_dl:
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            val_loss += Loss(Yhat, Y) * Y.shape[0]
            prediction += accuracy(Yhat, Y) # 
    return val_loss / prediction[1], prediction[0] / prediction[1]


def train(args):
    results, best_val_acc = [], 0.

    model = args.net().to(device)
    TrainLoader, ValLoader = args.get_data_function(args.batch_size, mode='train')
    loss_function = args.loss_function() # 

    # for param in model.ConvNets1.parameters():
    #     param.requires_grad = False
    # model.ConvNets1.load_state_dict(torch.load('./weights/v1/conv1_70ep_best.pkl'))

    if args.pretrained_weights is not None:
        print('train second time')
        model.load_state_dict(torch.load(args.pretrained_weights))
        # for name, param in model.named_parameters():
        #     if 'classifier' not in name:
        #         param.requires_grad_ = False
        _, best_val_acc = evaluate(model, ValLoader, loss_function)
        print(f'Initial Acc {best_val_acc:.6f}')
    else:
        print('train first time')

    optimizer = args.optimizer(model.parameters(), **args.kwargs_o)
    if args.scheduler is not None:
        scheduler = args.scheduler(optimizer, **args.kwargs_s)

    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        train_loss = 0.
        train_acc = torch.tensor([0, 0], dtype=int)
        lrs = []
        for X, Y in TrainLoader:
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            loss = loss_function(Yhat, Y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss * Yhat.shape[0]
            train_acc += accuracy(Yhat, Y)
            lrs.append(optimizer.param_groups[0]['lr'])
            
        if args.scheduler is not None:
            scheduler.step()

        t1 = time.time()

        epoch_train_acc = train_acc[0] / train_acc[1]
        epoch_train_loss = train_loss / train_acc[1]  
        epoch_val_loss, epoch_val_acc = evaluate(model, ValLoader, loss_function) 

        results.append({'avg_valid_loss': epoch_val_loss.item(),
                        'avg_val_acc': epoch_val_acc.item(),
                        'avg_train_loss': epoch_train_loss.item(),
                        'avg_train_acc': epoch_train_acc.item(),
                        'lrs': lrs})
        print(f'Epoch {epoch}  \tValid Loss {epoch_val_loss:.4f}\tTrain Loss {epoch_train_loss:.4f}', end='\t')
        print(f'Train Acc {epoch_train_acc:.4f}\tVal Acc {epoch_val_acc:.4f}', end="\t")
        print(f'Time {t1 - t0:.2f}s')

        torch.save(model, args.weights) 
        torch.save(model.state_dict(), args.weights_dict)
        if epoch_val_acc > best_val_acc:
            torch.save(model, args.best_weight) 
            torch.save(model.state_dict(), args.best_weight_dict)
            best_val_acc = epoch_val_acc

    Loss, Acc = evaluate(model, ValLoader, loss_function)
    if Acc > best_val_acc:
        torch.save(model, args.best_weight)
        torch.save(model.state_dict(), args.best_weight_dict)
        best_val_acc = Acc
    print(f'Total Loss {Loss:.6f}\tTotal Acc{Acc:.6f}\nBest Acc{best_val_acc:.6f}')
    if args.save_results: # 1e9 sec > 3 year, put last 8 digit of time as hash
        sslst = str(time.time()).split('.') 
        with open(f'{sslst[0][-8:]}_results.lst', 'wb') as file:
            pickle.dump(results, file)
    return results


def show_loss_acc(results, args, file=None, save=False):
    avg_train_loss, avg_val_loss = [], []
    avg_train_acc, avg_val_acc = [], []
    for result in results:
        avg_train_loss.append(result['avg_train_loss'])
        avg_val_loss.append(result['avg_valid_loss'])
        avg_train_acc.append(result['avg_train_acc'])
        avg_val_acc.append(result['avg_val_acc'])
    epochs_cnt = [i for i in range(args.epochs)]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_cnt, avg_train_loss, 'b', label="AvgTrainLoss", linewidth='2')
    plt.plot(epochs_cnt, avg_val_loss, 'green', label="AvgValLoss", linewidth='2')
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.ylim((0, 20)) # 
    plt.legend(fontsize=12)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_cnt, avg_train_acc, 'red', label="AvgTrainAcc", linewidth='2')
    plt.plot(epochs_cnt, avg_val_acc, 'black', label="AvgValAcc", linewidth='2')
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend(fontsize=12)
    if save:
        plt.savefig('./plot.png')
    plt.show()


def args_cnn():
    ap = argparse.ArgumentParser()
    epochs, batch_size, lr, = 50, 64, 0.03
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    # ap.add_argument('--validation_ratio', type=float, default=0.1)

    ap.add_argument('--net', default=models.cnn_cifar)
    ap.add_argument('--get_data_function', default=GetCIFAR100)

    ap.add_argument('--loss_function', default=nn.CrossEntropyLoss) # what's the difference with F.cross_entropy ?
    ap.add_argument('--optimizer', default=torch.optim.Adam)
    ap.add_argument('--kwargs_o', default={'lr': lr, 'weight_decay': 1e-5})
    ap.add_argument('--scheduler', default=torch.optim.lr_scheduler.OneCycleLR)
    ap.add_argument('--kwargs_s', default={'max_lr': lr, 'total_steps': 20 * 40000 // 64})

    ap.add_argument('--pretrained_weights', type=str, default=None)
    ap.add_argument('--weights', type=str, default='./weights/cnn.pt')
    ap.add_argument('--weights_dict', type=str, default='./weights/cnn_dict.pt')
    ap.add_argument('--best_weight', type=str, default='./weights/cnn_best.pt')
    ap.add_argument('--best_weight_dict', type=str, default='./weights/cnn_dict_best.pt')

    ap.add_argument('--save_results', type=bool, default=True)
    return ap.parse_args()


if __name__ == '__main__':
    Args = args_cnn()
    results = train(Args)
    show_loss_acc(results, Args)
