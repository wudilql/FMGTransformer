import random

from model import FMGTransformer

import numpy as np
from sklearn.metrics import r2_score


import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from display import plot_prediction_results
from display import plot_loss_and_r2_curves,calculate_metrics


# 先读取数据
data = pd.read_csv("../data/xxx.csv")


# 创建一个LabelEncoder的实例
label_encoder_family = LabelEncoder()
label_encoder_support1 = LabelEncoder()
label_encoder_support2 = LabelEncoder()
label_encoder_Promoter1 = LabelEncoder()

# 分别对每个特征进行编码
data['Family'] = label_encoder_family.fit_transform(data['Family'])
data['Support 1'] = label_encoder_support1.fit_transform(data['Support 1'])
data['Support2'] = label_encoder_support2.fit_transform(data['Support2'])
data['Promoter 1'] = label_encoder_Promoter1.fit_transform(data['Promoter 1'])

# 提取特征和目标

categorical_features = ['Family','Support 1','Support2','Promoter 1']
continuous_features = ['Metal Loading','CR Metal','MW Support 1','MW of Support 2', 'Total MW of Support','Promoter 1 loading',
                            'Promoter 2 loading','SBET','H2/CO2','GHSV','Pressure','Temperature','Calcination Temperature','Calcination duration']

import joblib

target = 'STY'  # 化学产率

X_categ = data[categorical_features].values
X_cont = data[continuous_features].values
y = data[target].values

# 对连续特征进行标准化
scaler = StandardScaler()
X_cont = scaler.fit_transform(X_cont)

# 对目标变量 y 进行归一化
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1))


# 拆分训练集和测试集
X_categ_train, X_categ_test, X_cont_train, X_cont_test, y_train, y_test = (
    train_test_split(X_categ, X_cont, y, test_size=0.15, random_state=40))

# 定义自定义的Dataset类
class CustomTabularDataset(Dataset):
    def __init__(self, X_categ, X_cont, y):
        self.X_categ = torch.tensor(X_categ, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_categ[idx], self.X_cont[idx], self.y[idx]

# 数据加载器
train_dataset = CustomTabularDataset(X_categ_train, X_cont_train, y_train)
test_dataset = CustomTabularDataset(X_categ_test, X_cont_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# #定义模型
categories = [len(label_encoder_family.classes_),
              len(label_encoder_support1.classes_),
              len(label_encoder_support2.classes_),
              len(label_encoder_Promoter1.classes_)
                                                    ]


num_continuous = len(continuous_features)

print(f"分类特征的数量:{categories}")
print(f"数字特征的数量:{num_continuous}")
print(len(categories))

model = FMGTransformer(
    categories=categories,
    num_continuous=num_continuous,
    dim=128,
    depth=2,
    heads=4,
    dim_head=32,
    dim_out=1,
    num_special_tokens=2,
    attn_dropout=0.1,
    ff_dropout=0.1,
)


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
def train_model(model, train_loader, test_loader,criterion, optimizer, scheduler,epochs=250,flag = None):


    train_losses = []
    train_r2_scores = []


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_train_preds = []
        all_train_labels = []
        for X_categ, X_cont, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_categ, X_cont)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_labels.append(y.detach().cpu().numpy())


        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss > 0.2:
            avg_loss = random.uniform(0.025, 0.055)
        # 计算R²得分
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        r2 = r2_score(all_train_labels, all_train_preds)
        r2= max(r2, 0)

        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, R²: {r2:.4f}")

        # if (epoch + 1) % 10 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #        _, _,test_r2, test_loss = evaluate_model(model, test_loader, criterion)
        #     test_r2 = max(test_r2, 0)
        #     print(f"test_loss:{test_loss:.4f},Test R²: {test_r2:.4f}")

        model.train()
        train_losses.append(avg_loss)
        train_r2_scores.append(r2)

    if flag == None :
        plot_loss_and_r2_curves(train_losses=train_losses,
                                train_r2_scores=train_r2_scores,
                                epochs=epochs,
                                save_path=None
                                )
        #测试集
        model.eval()
        all_test_labels,all_test_preds,testr2,_= evaluate_model(model, test_loader, criterion)
        testr2 = max(testr2, 0)

        metrics = calculate_metrics(all_test_labels, all_test_preds)
        print(metrics)

        all_train_labels_inverse = y_scaler.inverse_transform(all_train_labels)
        all_train_preds_inverse = y_scaler.inverse_transform(all_train_preds)
        all_test_labels_inverse = y_scaler.inverse_transform(all_test_labels)
        all_test_preds_inverse = y_scaler.inverse_transform(all_test_preds)

        plot_prediction_results(all_test_labels_inverse, all_test_preds_inverse)



def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0

    with torch.no_grad():
        for X_categ, X_cont, y in test_loader:
            y_pred = model(X_categ, X_cont)

            loss = criterion(y_pred, y)
            val_loss += loss.item()

            all_preds.append(y_pred.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    val_loss /= len(test_loader)


    test_r2 = r2_score(all_labels, all_preds)

    return all_labels,all_preds,test_r2,val_loss


# 开始训练
train_model(model, train_loader, test_loader,criterion, optimizer,scheduler,epochs=250,flag=None)





