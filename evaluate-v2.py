import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

n_gpus = torch.cuda.device_count()

iw = 24 * 1
ow = 1

class TFModel(nn.Module):
    def __init__(self, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
        return mask


    def forward(self, src, srcmask):
        print(src.shape)
        src = self.encoder(src)
        print(src.shape)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        L = y.shape[0]
        
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window

            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window

            Y[:, i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, 2))

        self.x = X
        self.y = Y

        self.len = len(X)

        plt.figure()

        plt.scatter([i for i in range(len(Y.flatten()))], Y.flatten())
        plt.ylim(-3.5, 3.5)
        plt.show()
        plt.savefig("./test.png")

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.len
    
def preprocessing():
    df = pd.read_excel("./HC5F0-5000 2 0.xlsx")

    X = pd.concat([df.loc[:][df.columns[-2:][0]], df.loc[:][df.columns[-2:][1]]], axis=1)

    dataTrain = X[:16681]
    dataTest = X[16681:]

    dataTrain.iloc[:, 0] = dataTrain.iloc[:, 0].index.values
    dataTrain.iloc[:, 0].astype('float64')

    dataTest.iloc[:, 0] = dataTest.iloc[:, 0].index.values
    dataTest.iloc[:, 0].astype('float64')

    XTrain = dataTrain[:len(dataTrain) - 1]
    TTrain = dataTrain[1:len(dataTrain)]

    XTest = dataTest[:len(dataTest) - 1]
    TTest = dataTest[1:len(dataTest)]

    std = StandardScaler()
    std.fit(XTrain)
    XTrain_scaled = std.transform(XTrain)
    XTest_scaled = std.transform(XTest)

    std.fit(TTrain)
    TTrain_scaled = std.transform(TTrain)
    TTest_scaled = std.transform(TTest)

    return [XTrain_scaled, XTest_scaled, std]
        
def MAPEval(y_pred, y_true):
    return mean_absolute_percentage_error(y_true, y_pred)

def MAE(y_pred, y_true):
    return mean_absolute_error(y_true, y_pred)

def RMSE(y_pred, y_true):
    return mean_squared_error(y_pred, y_true)**0.5

def evaluate(gpu, n_gpus, XTest_scaled, XTrain_scaled):
    num_worker = 8
    batch_size = 64
    
    model = TFModel(24*1, 1, 512, 8, 4, 0.1)
    model.load_state_dict(torch.load("./model-v2/back/model_%d.pth" % 145))

    model.to(gpu)

    model.eval()

    test_dataset = windowDataset(XTest_scaled[:, 1], input_window=iw, output_window=ow, stride=1)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_worker, sampler=test_sampler)

    train_dataset = windowDataset(XTrain_scaled[:, 1], input_window=iw, output_window=ow, stride=1)
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_worker, sampler=train_sampler)

    resultsl = None
    outputsl = None
    
    for i, (inputs, outputs) in enumerate(test_loader):
        inputs = inputs.to(gpu)
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(gpu)

        predictions = model(inputs.float(), src_mask)

        result = predictions.detach().cpu().float().numpy()

        if i == 0:
            resultsl = result.flatten()
            outputsl = outputs[:, :, 0].float().numpy().flatten()
        else:
            resultsl = np.concatenate((resultsl, result.flatten()))
            outputsl = np.concatenate((outputsl, outputs[:, :, 0].float().numpy().flatten()))

    previousl = None

    for i, (inputs, outputs) in enumerate(train_loader):
        if i == 0:
            previousl = outputs[:, :, 0].float().numpy().flatten()
        else:
            previousl = np.concatenate((previousl, outputs[:, :, 0].float().numpy().flatten()))


    print(MAPEval(resultsl, outputsl))
    print(MAE(resultsl, outputsl))
    print(RMSE(resultsl, outputsl))

    print("maximum of resultsl:", np.max(resultsl))
    print("minimum of resultsl:", np.min(resultsl))
    print("maximum of outputsl:", np.max(outputsl))
    print("minimum of resultsl:", np.min(outputsl))

    plt.figure()
    plt.plot([i for i in range(0, len(previousl) + len(outputsl))], np.append(previousl, outputsl), label="Actual", color='blue', linestyle='solid', marker=0, alpha=0.35)
    plt.plot([i for i in range(len(previousl), len(previousl) + len(resultsl))], resultsl, label="Prediction", color='red', linestyle='dashed', marker=0, alpha=0.25)
    plt.xlabel("index")
    plt.ylabel("Scaled fractional changes in resistivity")
    plt.ylim(-3.5, 3.5)
    plt.title("Prediction")
    plt.legend()

    plt.show()
    # plt.savefig("./result_%s.png" % "outputs")

    # # plt.figure()
    # plt.scatter([i for i in range(len(previousl))], previousl, color="black", label="Actual", marker=0, alpha=0.35)
    # plt.scatter([i for i in range(len(previousl), len(previousl) + len(outputsl))], outputsl, color="black", label="Ref Actual", marker=0, alpha=0.25)
    # plt.xlabel("index")
    # plt.ylabel("Scaled fractional changes in resistivity")
    # plt.title("Real")
    # plt.legend()

    # plt.show()
    plt.savefig("./result_%s.png" % "real")

if __name__ == "__main__":
    XTrain_scaled = preprocessing()[0]
    XTest_scaled = preprocessing()[1]
    std = preprocessing()[2]

    world_size = n_gpus
    
    ### Training
    # torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, XTrain_scaled))

    ### Test
    # torch.multiprocessing.spawn(evaluate, nprocs=n_gpus, args=(n_gpus, XTest_scaled, std))
    evaluate("cuda:1", "cuda:1", XTest_scaled, XTrain_scaled)


