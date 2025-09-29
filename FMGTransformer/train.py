import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from display import plot_prediction_results
from display import plot_loss_and_r2_curves,calculate_metrics



import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model import FTTransformer_MultiScaleGated




import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True  #


data = pd.read_csv("../data/xxx.csv")

label_encoder_family = LabelEncoder()
label_encoder_support1 = LabelEncoder()
label_encoder_support2 = LabelEncoder()
label_encoder_Promoter1 = LabelEncoder()
label_encoder_Promoter2 = LabelEncoder()
data['Family'] = label_encoder_family.fit_transform(data['Family'])
data['Support 1'] = label_encoder_support1.fit_transform(data['Support 1'])
data['Support2'] = label_encoder_support2.fit_transform(data['Support2'])
data['Promoter 1'] = label_encoder_Promoter1.fit_transform(data['Promoter 1'])
data['Promoter 2'] = label_encoder_Promoter2.fit_transform(data['Promoter 2'])

categorical_features = ['Family','Support 1','Support2','Promoter 1','Promoter 2']

continuous_features = ['Metal Loading','CR Metal', 'MW Support 1','MW of Support 2','Total MW of Support','Promoter 1 loading',
                            'Promoter 2 loading','SBET','H2/CO2','GHSV','Pressure','Temperature','Calcination Temperature','Calcination duration']


target = 'STY'

X_categ = data[categorical_features].values
X_cont_raw = data[continuous_features].values.astype('float32')
y_raw = data[target].values.reshape(-1, 1).astype('float32')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_categ_train, X_categ_test, X_cont_tr, X_cont_te, y_tr, y_te = train_test_split(
    X_categ, X_cont_raw, y_raw, test_size=0.15, random_state=40
)


x_scaler = StandardScaler().fit(X_cont_tr)
y_scaler = StandardScaler().fit(y_tr)

X_cont_train = x_scaler.transform(X_cont_tr)
X_cont_test  = x_scaler.transform(X_cont_te)
y_train      = y_scaler.transform(y_tr)
y_test       = y_scaler.transform(y_te)




class CustomTabularDataset(Dataset):
    def __init__(self, X_categ, X_cont, y):
        self.X_categ = torch.tensor(X_categ, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_categ[idx], self.X_cont[idx], self.y[idx]


train_dataset = CustomTabularDataset(X_categ_train, X_cont_train, y_train)
test_dataset = CustomTabularDataset(X_categ_test, X_cont_test, y_test)


use_cuda = (device.type == "cuda")

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    pin_memory=use_cuda, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=32,
    pin_memory=use_cuda, num_workers=0
)



categories = [len(label_encoder_family.classes_),
              len(label_encoder_support1.classes_),
              len(label_encoder_support2.classes_),
              len(label_encoder_Promoter1.classes_),
              len(label_encoder_Promoter2.classes_)
                                                    ]

num_continuous = len(continuous_features)


model = FTTransformer_MultiScaleGated(
    categories=categories,
    num_continuous=num_continuous,
    dim=160, depth=3, heads=5,
    attn_dropout=0.05, embed_dropout=0.05, token_dropout=0.0,
    pool="attn", use_cnn=False,
    numerical_mlp=True,
    head_dropout=0.0,
    fuser_residual=True, head_residual=True,
    drop_path=0.03,
    dim_out=1
).to(device)




criterion = nn.SmoothL1Loss(beta=0.07)

optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)



def train_model(model, train_loader,  val_loader, criterion, optimizer, epochs=250,flag = None):
    model.train()

    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(epochs):
        epoch_loss = 0
        all_train_preds = []
        all_train_labels = []
        for X_categ, X_cont, y in train_loader:
            X_categ = X_categ.to(device, non_blocking=True).long()
            X_cont = X_cont.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            y_pred = model(X_categ, X_cont)

            if y_pred.ndim == 1:
                y_pred = y_pred.unsqueeze(1)
            if y.ndim == 1:
                y = y.unsqueeze(1)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_labels.append(y.detach().cpu().numpy())

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        r2 = r2_score(all_train_labels, all_train_preds)
        r2= max(r2, 0)

        val_loss = 0
        val_r2,val_loss = evaluate_model(model, val_loader, criterion)
        val_r2 = max(val_r2, 0)

        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, R²: {r2:.4f}")
        print(f"val_loss:{val_loss:.4f},Test R²: {val_r2:.4f}")

        train_losses.append(avg_loss)
        train_r2_scores.append(r2)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)


    if flag == None :
        plot_loss_and_r2_curves(train_losses=train_losses,
            val_losses=val_losses,
            train_r2_scores=train_r2_scores,
            val_r2_scores=val_r2_scores,
            epochs=epochs,
            save_path=None
                            )


        all_test_preds = []
        all_test_labels = []
        with torch.no_grad():
            for X_categ, X_cont, y in test_loader:
                y_pred = model(X_categ, X_cont)
                all_test_preds.append(y_pred.detach().cpu().numpy())
                all_test_labels.append(y.detach().cpu().numpy())

        all_test_preds = np.concatenate(all_test_preds, axis=0)
        all_test_labels = np.concatenate(all_test_labels, axis=0)

        metrics = calculate_metrics(all_test_labels, all_test_preds)
        print(metrics)

        all_train_labels_inverse = y_scaler.inverse_transform(all_train_labels)
        all_train_preds_inverse = y_scaler.inverse_transform(all_train_preds)
        all_test_labels_inverse = y_scaler.inverse_transform(all_test_labels)
        all_test_preds_inverse = y_scaler.inverse_transform(all_test_preds)

        plot_prediction_results(all_test_labels_inverse, all_test_preds_inverse, train_losses)

        metrics = calculate_metrics(all_test_labels_inverse, all_test_preds_inverse)
        print(metrics)


def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0

    with torch.no_grad():
        for X_categ, X_cont, y in test_loader:
            X_categ = X_categ.to(device, non_blocking=True).long()
            X_cont = X_cont.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            y_pred = model(X_categ, X_cont)

            loss = criterion(y_pred, y)
            val_loss += loss.item()

            all_preds.append(y_pred.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    val_loss /= len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    all_preds = y_scaler.inverse_transform(all_preds)
    all_labels = y_scaler.inverse_transform(all_labels)

    test_r2 = r2_score(all_labels, all_preds)
    return test_r2,val_loss



train_model(model, train_loader, test_loader, criterion, optimizer,flag=None)
