#!/usr/bin/env python
# coding: utf-8

# # Deal or No Deal

# ### Dependency

# In[106]:


import os
import glob
import collections
import random

import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from datetime import datetime
from enum import Enum

import torch
from torch import nn
import torch.tensor
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.options.mode.chained_assignment = None

# ### Configs

# In[107]:


TRAINING_DATE_FROM = datetime.strptime('1990-03-26', '%Y-%m-%d')
TRAINING_DATE_TO = datetime.strptime('2019-07-01', '%Y-%m-%d')

VALIDATION_DATE_FROM = datetime.strptime('1990-03-26', '%Y-%m-%d')
VALIDATION_DATE_TO = datetime.strptime('2019-12-31', '%Y-%m-%d')

TESTING_DATE_FROM = datetime.strptime('1990-03-26', '%Y-%m-%d')
TESTING_DATE_TO = datetime.strptime('2020-02-13', '%Y-%m-%d')

PREDICT_UP_TO = 21

# In[108]:


DATA_DIR = '.'
REGION = 'us'
INSTRUMENT = 'msft'

RAW_DATA_FULL_FEATURES_SET = ['date', 'open', 'high', 'low', 'close', 'volume', 'openint']
RAW_DATA_REMOVED_FEATURES_SET = ['open', 'high', 'low', 'close']
RAW_DATA_ADDED_FEATURES_SET = ['volume']
RAW_DATA_FEATURES_SET = RAW_DATA_REMOVED_FEATURES_SET + RAW_DATA_ADDED_FEATURES_SET

RAW_DATA_FEATURES_SET = ['open', 'high', 'low', 'close', 'volume']
RAW_TARGET = 'close'

FEATURES_SET = ['open',
                'high',
                'low',
                'close']
TARGET = 'close_delta'


# ## Helpers

# ### Market Data

# In[109]:


def fetch_instrument_file(instrument, region, base_dir):
    cwd = os.getcwd()
    os.chdir(base_dir)

    instrument_file_list = result = list(Path(".").rglob(f"*{instrument}*.txt"))
    if not instrument_file_list:
        os.chdir(cwd)
        raise Exception(f'Cannot find file for instrument {instrument}')
    if len(instrument_file_list) > 1:
        os.chdir(cwd)
        raise Exception(f'Found multiple file for instrument {instrument}')

    os.chdir(cwd)
    return str(os.path.join(os.getcwd(), instrument_file_list[0]))


instrument_data_file = fetch_instrument_file(INSTRUMENT, REGION, DATA_DIR)


# # Data Preparation

# In[110]:


instrument_data_file = fetch_instrument_file(INSTRUMENT, REGION, DATA_DIR)

# In[111]:


df = pd.read_csv(instrument_data_file)
df.columns = RAW_DATA_FULL_FEATURES_SET
df.date = pd.to_datetime(df.date, format='%Y%m%d')

# #### Split dataset - (Training / Validation / Testing)

# In[112]:


training_df = df[(df['date'] >= TRAINING_DATE_FROM) & (df['date'] <= TRAINING_DATE_TO)]
training_df = training_df.reset_index()
training_df.drop(['index'], axis=1, inplace=True)
training_df = training_df.iloc[::-1]

validation_df = df[(df['date'] >= VALIDATION_DATE_FROM) & (df['date'] <= VALIDATION_DATE_TO)]
validation_df = validation_df.reset_index()
validation_df.drop(['index'], axis=1, inplace=True)
validation_df = validation_df.iloc[::-1]

testing_df = df[(df['date'] >= TESTING_DATE_FROM) & (df['date'] <= TESTING_DATE_TO)]
testing_df = testing_df.reset_index()
testing_df.drop(['index'], axis=1, inplace=True)
testing_df = testing_df.iloc[::-1]

# # Data Preparation /Training

# #### Trim dataframe

# In[113]:


training_df = training_df[RAW_DATA_FEATURES_SET]


# In[114]:


class features_extraction_ops(Enum):
    calc = 'calc'
    clean = 'clean'
    statistics = 'statistics'
    standardize = 'standardize'


def features_extraction(df, ops=[]):
    def _calc(df):
        df[f'{RAW_TARGET}_t1'] = df[RAW_TARGET].shift(-1)
        df[TARGET] = df[f'{RAW_TARGET}_t1'] - df[RAW_TARGET]

        df = df[:-1]

        return df

    def _clean(df):
        df.drop([f'{RAW_TARGET}_t1'], axis=1, inplace=True)

        return df

    def _statistics(df):
        return df.describe()

    def _standardize(df):
        for feature in FEATURES_SET:
            df[feature] = MinMaxScaler().fit_transform(df[[feature]])
        df[TARGET] = MinMaxScaler().fit_transform(df[[TARGET]])

        return df

    if features_extraction_ops.calc in ops:
        df = _calc(df)
    elif features_extraction_ops.clean in ops:
        df = _clean(df)
    elif features_extraction_ops.statistics in ops:
        df = _statistics(df)
    elif features_extraction_ops.standardize in ops:
        df = _standardize(df)

    return df


training_df = features_extraction(training_df, [features_extraction_ops.calc])
training_df = features_extraction(training_df, [features_extraction_ops.clean])
training_statistics_df = features_extraction(training_df, [features_extraction_ops.statistics])
training_df = features_extraction(training_df, [features_extraction_ops.standardize])


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _get_ops(op):
    if op == 'avg':
        return np.mean

    raise Exception(f'Unknown operation {op}')


def _aggregate_collection(lst, op):
    if len(lst) == 0:
        return {}
    elif len(lst) == 1:
        return lst[0]
    else:
        aggrgated_values_dict = {}
        for entry in lst:
            for key, value in entry.items():
                if key in aggrgated_values_dict:
                    aggrgated_values_dict[key].append(value)
                else:
                    aggrgated_values_dict[key] = [value]

        final_aggrgated_values_dict = {}
        for key, values in aggrgated_values_dict.items():
            values = [e for e in values if e]
            final_aggrgated_values_dict[key] = op(values)

        return final_aggrgated_values_dict


def _day_aggregator_handler(dataset, **kwargs):
    length = kwargs['length']
    return dataset[0:length], dataset[length:]


def _week_aggregator_handler(dataset, **kwargs):
    return _universal_aggregator_handler(dataset, 5, **kwargs)


def _month_aggregator_handler(dataset, **kwargs):
    return _universal_aggregator_handler(dataset, 21, **kwargs)


def _quarterly_aggregator_handler(dataset, **kwargs):
    return _universal_aggregator_handler(dataset, 21 * 3, **kwargs)


def _yearly_aggregator_handler(dataset, **kwargs):
    return _universal_aggregator_handler(dataset, 21 * 12, **kwargs)


def _universal_aggregator_handler(dataset, period_range, **kwargs):
    length = kwargs['length']
    ops = _get_ops(kwargs['op'])

    if length == -1:
        return [_aggregate_collection(dataset, ops)], []

    else:
        aggregated_dataset = []

        raw_dataset = dataset[0:length]
        raw_chunked_dataset = chunks(raw_dataset, period_range)
        for week in raw_chunked_dataset:
            aggregated_dataset.append(_aggregate_collection(week, ops))

        return aggregated_dataset, dataset[length:]


def features_aggregator(dataset, features_descriptors):
    aggregated_dataset = []
    for feature_set in features_descriptors:
        index, length, handler, params = feature_set['index'], feature_set['length'], feature_set['handler'], feature_set['params']

        if length == -1:
            period_range = len(dataset)
            step_aggregated_dataset, dataset = handler(dataset, period_range, **{
                'index': index,
                'length': length,

                **params
            })
        else:
            step_aggregated_dataset, dataset = handler(dataset, **{
                'index': index,
                'length': length,

                **params
            })

        if len(step_aggregated_dataset) > 0:
            aggregated_dataset += step_aggregated_dataset

        if len(dataset) == 0:
            return aggregated_dataset

    return aggregated_dataset


def train_dataset_generator(df, shift_range, repeat_out):
    _features_descriptors = [
        {'index': 0, 'length': 200, 'handler': _day_aggregator_handler, 'params': {}},
        # {'index': 1, 'length': 21 * 12 * 6, 'handler': _week_aggregator_handler, 'params': {'op': 'avg'}},
        # {'index': 2, 'length': 21 * 12 * 6, 'handler': _month_aggregator_handler, 'params': {'op': 'avg'}},
        # {'index': 3, 'length': 21 * 12 * 6, 'handler': _quarterly_aggregator_handler, 'params': {'op': 'avg'}},
        # {'index': 4, 'length': 21 * 12 * 6, 'handler': _yearly_aggregator_handler, 'params': {'op': 'avg'}},
        # {'index': 5, 'length': -1, 'handler': _universal_aggregator_handler, 'params': {'op': 'avg'}},
    ]

    dataset = list(df.T.to_dict().values())
    train_dataset = []
    while repeat_out > 0:
        train_subset = features_aggregator(dataset, _features_descriptors)
        train_dataset.append(train_subset)
        dataset = dataset[shift_range:]
        repeat_out -= 1

    return train_dataset


train_dataset = train_dataset_generator(training_df, shift_range=1, repeat_out=100)


def pad_dataset_sequence(dataset):
    if len(dataset) == 0:
        return dataset

    max_sequence = max([len(e) for e in dataset])

    input_example_dict = dataset[0][0]
    pad_example_dict = {k: 0.0 for k, _ in input_example_dict.items()}

    for example in dataset:
        residual_sequence_length = max_sequence - len(example)
        if residual_sequence_length > 0:
            for e in [pad_example_dict] * residual_sequence_length:
                example.append(e)

    return dataset


print(collections.Counter([len(e) for e in train_dataset]))
train_dataset = pad_dataset_sequence(train_dataset)
print(collections.Counter([len(e) for e in train_dataset]))
# random.shuffle(train_dataset)


class HyperParams:
    class training:
        epochs = 50
        batch_size = 1
        learning_rate = 0.01

    class Model:
        class RNN:
            input_size = len(FEATURES_SET)
            sequence_size = len(train_dataset[0])
            layers_size = 1
            hidden_size = 1
            dropout_rate = 0.05

        class Dense:
            layer_1 = 256
            layer_2 = 512
            layer_3 = 256

            dropout = 0.05

            input_size = len(train_dataset[0])
            output_size = 1


gpu_enabled = torch.cuda.is_available()
device = torch.device("cuda") if gpu_enabled else torch.device("cpu")

class SingleInstrumentPredictorRNN(nn.Module):

    def __init__(self):
        super(SingleInstrumentPredictorRNN, self).__init__()

        self._rnn = nn.LSTM(input_size=HyperParams.Model.RNN.input_size,
                            hidden_size=HyperParams.Model.RNN.hidden_size,
                            num_layers=HyperParams.Model.RNN.layers_size,
                            batch_first=True)

        self._fc_1 = nn.Linear(HyperParams.Model.Dense.input_size, HyperParams.Model.Dense.layer_1)
        self._fc_2 = nn.Linear(HyperParams.Model.Dense.layer_1, HyperParams.Model.Dense.layer_2)
        self._fc_3 = nn.Linear(HyperParams.Model.Dense.layer_2, HyperParams.Model.Dense.layer_3)
        self._fc_4 = nn.Linear(HyperParams.Model.Dense.layer_3, HyperParams.Model.Dense.output_size)
        self._drop_layer = nn.Dropout(p=HyperParams.Model.Dense.dropout)

    def forward(self, input):
        batch_size = input.size(0)

        hidden = self.init_hidden(batch_size)
        out, hidden = self._rnn(input.double(), hidden)
        out = out.view(-1, HyperParams.Model.Dense.input_size)

        # out = (F.relu(self._fc_1(out)))
        # out = (F.relu(self._fc_2(out)))
        # out = (F.relu(self._fc_3(out)))
        # out = self._fc_4(out)

        return hidden[0].squeeze(1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(HyperParams.Model.RNN.layers_size,
                             batch_size,
                             HyperParams.Model.RNN.hidden_size).zero_().double().to(device),
                  weight.new(HyperParams.Model.RNN.layers_size,
                             batch_size,
                             HyperParams.Model.RNN.hidden_size).zero_().double().to(device))

        return hidden

model = SingleInstrumentPredictorRNN().double()
print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=HyperParams.training.learning_rate)

if gpu_enabled:
    model.cuda()
    print('GPU Enabled Model')
else:
    print('GPU Disabled Model')


def tensorify_example(example):
    example_df = pd.DataFrame(example)

    features_tensor = torch.tensor(example_df[FEATURES_SET].values, dtype=torch.double)
    features_tensor = features_tensor.unsqueeze(0)

    target_tensor = torch.tensor(example_df[TARGET].values[0], dtype=torch.double)
    target_tensor = target_tensor.view(1, -1)

    #  features_tensor.size() # torch.Size([1, 658, 7])
    # target_tensor.size() # torch.Size([1, 658])
    return features_tensor, target_tensor


def batch_tensorify(examples_batch):
    features_tensors_list = [tensorify_example(example)[0] for example in examples_batch]
    target_tensors_list = [tensorify_example(example)[1] for example in examples_batch]

    return torch.cat(features_tensors_list, 0), torch.cat(target_tensors_list, 0)



def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


for epoch in range(1, HyperParams.training.epochs + 1):
    epoch_loss = 0.0
    mini_loss = 10 ^ 100
    for batch_examples in batch(train_dataset, HyperParams.training.batch_size):
        batch_features, batch_target = batch_tensorify(batch_examples)
        batch_features, batch_target = batch_features.double(), batch_target.double()

        if gpu_enabled:
            batch_features = batch_features.cuda()
            batch_target = batch_target.cuda()

        output = model(batch_features.double())

        loss = criterion(output, batch_target)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    if epoch_loss < mini_loss:
        mini_loss = epoch_loss
        torch.save(model.state_dict(), f'./{INSTRUMENT}_{HyperParams.training.epochs}epochs_miniloss_model_state.pt')

    print(f'Epoch: {epoch}/{HyperParams.training.epochs} ............. Loss: {epoch_loss / HyperParams.training.epochs}')

torch.save(model.state_dict(), f'./{INSTRUMENT}_{HyperParams.training.epochs}epochs_model_state.pt')


# # Evaludation

# #### Load model

model = SingleInstrumentPredictorRNN().double()
if gpu_enabled:
    model.cuda()
    print('GPU Enabled Model')
else:
    print('GPU Disabled Model')

model.load_state_dict(torch.load(f'./{INSTRUMENT}_{HyperParams.training.epochs}epochs_model_state.pt'))
model.eval()


testing_df = testing_df[RAW_DATA_FEATURES_SET]

testing_df = features_extraction(testing_df, [features_extraction_ops.calc])
testing_df = features_extraction(testing_df, [features_extraction_ops.clean])
testing_statistics_df = features_extraction(testing_df, [features_extraction_ops.statistics])
testing_df = features_extraction(testing_df, [features_extraction_ops.standardize])


testing_dataset = train_dataset_generator(testing_df, shift_range=1, repeat_out=14)

preidcted_list = []
expected_list = []
for test_example in testing_dataset:
    batch_features, batch_target = batch_tensorify([test_example])
    batch_features, batch_target = batch_features.double(), batch_target.double()

    print('batch_features ', batch_features)
    print('batch_target ', batch_target)
    if gpu_enabled:
        batch_features = batch_features.cuda()
        batch_target = batch_target.cuda()

    with torch.no_grad():
        output = model(batch_features)

    print('output ', output)
    preidcted_list.append(output[0][0])
    expected_list.append(batch_target[0][0])

    print('---------------------------------- ')
    print('')
print(preidcted_list)
print(expected_list)

