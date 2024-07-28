import os
from numpy import asarray, sqrt
import torch
import models
from PIL.Image import open as Open
import argparse
'''
undone
'''
model = Net()
model.load_state_dict(torch.load('./weights/cnn_n_70ep_49.pkl', map_location=torch.device('cpu')))


def move_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return [move_device(element, device) for element in tensor]
    return tensor.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dataloader, device='cpu'):
        self.dl = dataloader
        self.device = device

    def __iter__(self):
        for i in self.dl:
            yield move_device(i, self.device)

    def __len__(self):
        return len(self.dl)


def load_test_data():
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, pin_memory=True)
    test_dl = DeviceDataLoader(test_loader)
    return test_dl

def accuracy(predicted, labels):
    pred, predclassid = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predclassid == labels).item() / len(predicted))


def evaluate(model, valid_dl, loss_func):
    model.eval()
    loss_per_batch, accuracy_per_batch = [], []
    for images, labels in valid_dl:
        with torch.no_grad():
            predicted = model(images)
        loss_per_batch.append(loss_func(predicted, labels))
        accuracy_per_batch.append(accuracy(predicted, labels))
    val_loss_epoch = torch.stack(loss_per_batch).mean().item()
    val_accuracy_epoch = torch.stack(accuracy_per_batch).mean().item()
    return val_loss_epoch, val_accuracy_epoch


def show_prediction():
    file, img = [], []
    directory = './image_show'
    bijection = {}
    with open('./image_show/son/fine_label_names.txt', mode='r') as f:
        for line in f.readlines():
            [idx, name] = line.split(' ')
            bijection[idx] = name
    for item in os.walk(directory, topdown=True):
        for it in item:
            if type(it) == list:
                file.extend(it)
            elif type(it) == str:
                file.append(it)
    for f in file:
        if f.find('.png') != -1:
            img.append(os.path.join(directory, f))

    for image in img:
        X = torch.tensor(asarray(Open(image))).float()
        X = X.permute(2, 0, 1)
        X = X.reshape(-1, 3, 32, 32)
        prediction = model(X).flatten()
        id_ = int(prediction.argmax())
        print(f'{image}\t{bijection[f"{id_}"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', type=str, help='evaluate or predict')
    args = vars(ap.parse_args())
    if args["mode"] == 'evaluate':
        from data_loader import test_data
        test_dl = load_test_data()
        _, best_acc = evaluate(model, test_dl, torch.nn.functional.cross_entropy)
        print(f'Acc {best_acc:.6f}')
    else:
        show_prediction()
