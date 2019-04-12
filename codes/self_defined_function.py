import pandas as pd
import shapely
import numpy as np
import sys
import torch
import argparse
import random
import math
import os
import warnings
from datetime import datetime
from gensim.models import word2vec
from shapely.geometry import LineString, Polygon
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler# 好处在于可以保存训练集中的参数（均值、方差）
from scipy.stats import stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import tqdm
import matplotlib.pyplot as plt

def preprocessing(df, has_columns = False):
    # set columns name
    if not has_columns:
        columns = ["Bew_Typ", "BewSchl", "BewTxt", "BewSchl2", "LagerOrt", "Art_Typ", "ArtGrp", "ArtNr", "Bew_Datum", "Bew_Jahr", "Bew_Monat", "Menge", "EinheitNr", "Einheit", "Lief_Kd_Nr", "Best_Auft_Nr"]
        df.columns = columns
    df = df[df['Bew_Jahr'] >= 2005][['Bew_Typ', 'Bew_Datum', 'Bew_Jahr', 'ArtNr', 'ArtGrp', 'Menge', 'Lief_Kd_Nr']]
    df = df[df['Bew_Typ'] == 'Verkauf']
    df = df[df['ArtGrp'] < 20]
    df['ArtNr'] = df['ArtNr'].map(lambda x: str(x)[:6])
    df['Bew_Datum'] = pd.to_datetime(df['Bew_Datum'], dayfirst=True)
    #df['Menge'] = df['Menge'].map(lambda x: x.replace(',', '.'))
    #df['Menge'] = pd.to_numeric(df['Menge'])  
    df = df[df['Menge'] < 0]
    df = df.drop_duplicates()
    df['Menge'] = -df['Menge']
    df['weekday'] = df['Bew_Datum'].map(lambda x: x.weekday())
    return df

def create_grid_data_for_an_article(ArtNr, df, output = '../results/df_200.pkl'):
    """
    Attention:
        - df should contain feature: "Bew_Datum", "Menge", "ArtNr"
        - start date is 2015-01-01, finish date is 2019-03-29. They are fixed
    """
    # create empty grid
    START_TIME = datetime(2005, 1, 1)
    END_TIME = datetime(2019, 3, 29)
    index_dax = pd.date_range(START_TIME, END_TIME,freq='D').date
    df_grid = pd.DataFrame(index = index_dax)
    # choose the target article and agg the menge
    df_200 = df[df['ArtNr'] == ArtNr][['Bew_Datum', 'Menge']].groupby('Bew_Datum').sum()
    # join two table
    df_200 = df_grid.join(df_200, how = 'left')
    # fill null with 0
    df_200 = df_200.fillna(0)
    # save the new dataframe
    pd.to_pickle(df_200, output)
    return df_200

def extract_dataset_from_timeseries(df, targetname = 'Menge', lookback = 4):
    """
    df should not contain the features, which will not used in the traning of the model
    """
    cols = df.columns.to_list()
    cols_x = [i for i in cols if i not in [targetname]]
    cols_y = [targetname]
    for i in range(lookback):
        df[targetname+'_'+str(i+1)] = df[targetname].shift(i+1)
    dat_x = df[cols_x].iloc[lookback:, :].values
    dat_y = df[cols_y].iloc[lookback:, :].values
    return dat_x, dat_y




##################
# model relevant #
##################
class TS_rnn(torch.nn.Module):
    """
    scores for each piece
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size, seq_len)
    """
    def __init__(self, num_inp = 13, num_hidden = 64, num_layers = 2, dropout = 0.5, num_dim_mlp = 16):
        super(TS_rnn, self).__init__()
        #change the structure of the network
        num_inp = num_inp
        self.rnn = torch.nn.LSTM(input_size = num_inp, hidden_size = num_hidden, num_layers = num_layers, dropout = dropout)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(num_hidden, num_dim_mlp),
                torch.nn.Dropout(),
                torch.nn.ReLU(),
                torch.nn.Linear(num_dim_mlp, 1)
                )

    def forward(self, inp):
        # input of the rnn (seq_len, batch, input_size)
        data_in = torch.transpose(inp, 0, 1)
        # run rnn, it has two output
        out_rnn, _ = self.rnn(data_in)
        out_rnn = torch.transpose(out_rnn, 0, 1) # (batch_size, seq_len, num_dim)
        # rnn the mlp
        out = self.mlp(out_rnn[:, -1, :])
        # rnn the mlp
        #batch_size, seq_len, num_dim = out_rnn.shape
        #out = []
        #for i in range(seq_len):
        #    tmp = self.mlp(out_rnn[:, i,:])
        #    out.append(tmp)
        #now out is list of (batch_size, 1), combine the items in the list to get the output with size (batch_size, seq_len)
        #out = torch.cat(out, 1)
        return out

class MLP(torch.nn.Module):
    def __init__(self, num_inp = 1, num_hidden = 16, num_hidden2 = 0):
        super(MLP, self).__init__()
        #change the structure of the network
        num_inp = num_inp
        if num_hidden2 == 0:
            self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(num_inp, num_hidden),
                    torch.nn.BatchNorm1d(num_hidden),
                    torch.nn.ReLU(),
                    #torch.nn.Linear(num_hidden, num_hidden2),
                    #torch.nn.BatchNorm1d(num_hidden2),
                    #torch.nn.PReLU(),
                    torch.nn.Linear(num_hidden, 1)
                    )
        else:
            self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(num_inp, num_hidden),
                    torch.nn.BatchNorm1d(num_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_hidden, num_hidden2),
                    torch.nn.BatchNorm1d(num_hidden2),
                    torch.nn.PReLU(),
                    torch.nn.Linear(num_hidden2, 1)
                    )
    def forward(self, inp):
        out = self.mlp(inp)
        return out
    
class Data:
    """
    data class for TS_rnn
    """
    def __init__(self, x, y):
        self.data = {}
        self.data['train_x'] = self.add_file(x).float()
        #self.data['train_y'] = self.add_file(y).float()
        self.data['train_y'] = self.add_file(y).squeeze().float() 
        #self.data['train_y'] = self.add_file(y)[:, -1, :].squeeze().float() 
        #相比于上一行的target，这一行的target的表现要差了很多，为什么，这一行的不是才是正确的吗？？？？？？？？？？？？？？？
        #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, path):
        return torch.from_numpy(np.load(path))

    def add_scores(self, path):
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['train_x'][index],
                self.data['train_y'][index])
# write the test function
def test_model(dl_test, model, loss):
    model.eval()
    test_loss = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        # codes to be changed
        inp, target = dat
        out = model(inp)
        lo = loss(out, target)
        test_loss += lo.data
    return test_loss/counter