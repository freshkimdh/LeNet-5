# Le-Net 5 구현하기
# https://deep-learning-study.tistory.com/503

# %%
from torchvision import transforms
from torchvision import datasets # # MNIST training dataset 불러오기
from torch.utils.data import DataLoader
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import torch

# transformation 정의하기
data_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    ])

# 데이터를 저장할 경로 설정
path_data = '/DL_ALL'

# training data 불러오기
train_data = datasets.MNIST(path_data, train=True, download=True, transform=data_transform)

# MNIST test dataset 불러오기
val_data = datasets.MNIST(path_data, train=False, download=True, transform=data_transform)

# cuda 사용 가능 확인
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)



# %%

# training data를 추출합니다.
x_train, y_train = train_data.data, train_data.targets

# val data를 추출합니다.
x_val, y_val = val_data.data, val_data.targets

# 차원을 추가하여 B*C*H*W 가 되도록 합니다.
if len(x_train.shape) == 3:
    x_train = x_train.unsqueeze(1)
    print(x_train.shape)

if len(x_val.shape) == 3:
    x_val = x_val.unsqueeze(1)

# tensor를 image로 변경하는 함수를 정의합니다.
def show(img):
    # tensor를 numpy array로 변경합니다.
    npimg = img.numpy()
    # C*H*W를 H*W*C로 변경합니다.
    npimg_tr = npimg.transpose((1,2,0))
    plt.imshow(npimg_tr, interpolation='nearest')

# images grid를 생성하고 출력합니다.
# 총 40개 이미지, 행당 8개 이미지를 출력합니다.
x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)

show(x_grid)

# %%
# data loader 를 생성합니다.
from torch.utils.data import DataLoader

train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
val_dl = DataLoader(val_data, batch_size=32)

# %%

from torch import nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1) # in_channels, out_channels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        print('x_0 = ',x.shape)
        x = F.tanh(self.conv1(x))
        print('x_1 = ',x.shape)
        x = F.avg_pool2d(x, 2, 2)
        print('x_2 = ',x.shape)
        x = F.tanh(self.conv2(x))
        print('x_3 = ',x.shape)
        x = F.avg_pool2d(x, 2, 2)
        print('x_4 = ',x.shape)
        x = F.tanh(self.conv3(x))
        print('x_5 = ',x.shape)
        x = x.view(-1, 120)
        print('x_6 = ',x.shape)
        x = F.tanh(self.fc1(x))
        print('x_7 = ',x.shape)
        x = self.fc2(x)
        print('x_8 = ',x.shape)
        return F.softmax(x, dim=1)

model = LeNet_5()
print(model)
# %%

# 모델을 cuda로 전달
model.to(device)
print(next(model.parameters()).device)

# %%
# 파라미터
# 파라미터는 모델 내부에서 결정되는 변수

# 하이퍼 파라미터
# 하이퍼 파라미터는 모델링할 때 사용자가 직접 세팅해주는 값
# https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-13-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0Parameter%EC%99%80-%ED%95%98%EC%9D%B4%ED%8D%BC-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0Hyper-parameter

# 모델 summary를 확인
from torchsummary import summary
summary(model, input_size=(1, 32, 32))

# %%
# loss function 정의합니다.
loss_func = nn.CrossEntropyLoss(reduction='sum')


# %%
# optimizer 정의합니다.
from torch import optim
opt = optim.Adam(model.parameters(), lr=0.001)

# 현재 lr을 계산하는 함수를 정의합니다.
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    
# 러닝레이트 스케쥴러를 정의합니다.
from torch.optim.lr_scheduler import CosineAnnealingLR
lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-05)


# %%
# 배치당 performance metric 을 계산하는 함수 정의
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# %%
# 배치당 loss를 계산하는 함수를 정의
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

# %%
# epoch당 loss와 performance metric을 계산하는 함수 정의
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.type(torch.float).to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True: # sanity_check가 True이면 1epoch만 학습합니다.
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric


# %%
# train_val 함수 정의
import copy
import os

def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {
        'train': [],
        'val': [],
    }

    metric_history = {
        'train': [],
        'val': [],
    }

    # best model parameter를 저장합니다.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights')

        lr_scheduler.step()

        print('train loss: %.6f, dev loss: %.6f, accuracy: %.2f' %(train_loss, val_loss, 100*val_metric))
        print('-'*10)

    # best model을 반환합니다.
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

# %%
# 학습된 모델의 가중치를 저장할 폴더를 만듭니다.
os.makedirs('/models', exist_ok=True)

# 하이퍼 파라미터를 설정합니다.
params_train={
 "num_epochs": 3,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_dl,
 "val_dl": val_dl,
 "sanity_check": False,
 "lr_scheduler": lr_scheduler,
 "path2weights": "/models/LeNet-5.pt",
}

# %%
# 모델을 학습합니다.
model, loss_hist, metric_hist = train_val(model,params_train)

# %%
# plot loss progress
num_epochs = params_train["num_epochs"]

plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
# %%

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
# %%
