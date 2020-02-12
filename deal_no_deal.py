#!/usr/bin/env python
import collections
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

DATA_DIR = './data'
RESOLUTION = 'daily'
REGION = 'us'
INSTRUMENT = 'unh'
INSTRUMENT_PATH = './nyse stocks/2/unh.us.txt'

raw_data_features_set = ['open', 'high', 'low', 'close']
raw_target = 'close'

features_set = ['volume',
                'high_open_delta',
                'low_close_delta',
                'high_low_delta',
                'open_close_delta',
                'high_close_delta',
                'low_open_delta']
target = ['close_delta']

data_file = os.path.normpath(os.path.join(DATA_DIR, RESOLUTION, REGION, INSTRUMENT_PATH))

df = pd.read_csv(data_file)
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'openint']

df.date = pd.to_datetime(df.date, format='%Y%m%d')

df['high_open_delta'] = df['high'] - df['open']
df['low_close_delta'] = df['low'] - df['close']

df['high_low_delta'] = df['high'] - df['low']
df['open_close_delta'] = df['open'] - df['close']

df['high_close_delta'] = df['high'] - df['close']
df['low_open_delta'] = df['low'] - df['open']

df[f'{raw_target}_t1'] = df[raw_target].shift(-1)
df[f'{raw_target}_delta'] = df[f'{raw_target}_t1'] - df[raw_target]

for feature in features_set:
    df[feature] = StandardScaler().fit_transform(df[[feature]])
df[f'{raw_target}_delta'] = StandardScaler().fit_transform(df[[f'{raw_target}_delta']])

df.drop([f'{raw_target}_t1'], axis=1, inplace=True)
df.drop(raw_data_features_set + ['date', 'openint'], axis=1, inplace=True)
df = df[:-1]


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


def train_dataset_generator(df, shift_range=21, repeat_out=2):
    _features_descriptors = [
        {'index': 0, 'length': 21 * 12 * 1, 'handler': _day_aggregator_handler, 'params': {}},
        {'index': 1, 'length': 21 * 12 * 6, 'handler': _week_aggregator_handler, 'params': {'op': 'avg'}},
        {'index': 2, 'length': 21 * 12 * 6, 'handler': _month_aggregator_handler, 'params': {'op': 'avg'}},
        {'index': 3, 'length': 21 * 12 * 6, 'handler': _quarterly_aggregator_handler, 'params': {'op': 'avg'}},
        {'index': 4, 'length': 21 * 12 * 6, 'handler': _yearly_aggregator_handler, 'params': {'op': 'avg'}},
        {'index': 5, 'length': -1, 'handler': _universal_aggregator_handler, 'params': {'op': 'avg'}},
    ]

    dataset = list(df.T.to_dict().values())
    train_dataset = []
    while repeat_out > 0:
        train_subset = features_aggregator(dataset, _features_descriptors)
        train_dataset.append(train_subset)
        dataset = dataset[shift_range:]
        repeat_out -= 1

    return train_dataset


train_dataset = train_dataset_generator(df, shift_range=5, repeat_out=100)


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

# Hyper-params

param_epochs = 200
param_batch_size = 1

param_input_size = 7
param_sequence_size = 658
param_layers_size = 1
param_hidden_size = 1
param_dropout = 0.05
param_dense_1 = 1024
param_dense_2 = 512
param_dense_3 = 128
param_output_size = 1

gpu_enabled = torch.cuda.is_available()
device = torch.device("cuda") if gpu_enabled else torch.device("cpu")

param_lr = 0.01


class SingleInstrumentPredictorRNN(nn.Module):

    def __init__(self):
        super(SingleInstrumentPredictorRNN, self).__init__()

        self._rnn = nn.RNN(param_input_size, param_hidden_size, param_layers_size, batch_first=True)

        self._fc_1 = nn.Linear(param_sequence_size, param_dense_1)
        self._fc_2 = nn.Linear(param_dense_1, param_dense_2)
        self._fc_3 = nn.Linear(param_dense_2, param_dense_3)
        self._fc_4 = nn.Linear(param_dense_3, param_sequence_size)

    def forward(self, input):
        batch_size = input.size(0)

        hidden = self.init_hidden(batch_size)
        out, hidden = self._rnn(input.double(), hidden.double())

        out = out.view(-1, param_sequence_size)

        out = F.relu(self._fc_1(out))
        out = F.relu(self._fc_2(out))
        out = F.relu(self._fc_3(out))
        out = self._fc_4(out)

        return out, hidden

    def init_hidden(self, batch_size):
        return torch.rand(param_layers_size, batch_size, param_hidden_size, dtype=torch.double)


model = SingleInstrumentPredictorRNN()
model = model.double()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=param_lr)

if gpu_enabled:
    model.cuda()
    print('GPU Enabled Model')
else:
    print('GPU Disabled Model')


def tensorify_example(example):
    example_df = pd.DataFrame(example)

    features_tensor = torch.tensor(example_df[features_set].values, dtype=torch.double)
    features_tensor = features_tensor.unsqueeze(0)

    target_tensor = torch.tensor(example_df[target].values, dtype=torch.double)
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


for epoch in range(1, param_epochs + 1):

    loss = 0
    for batch_examples in batch(train_dataset, param_batch_size):

        batch_features, batch_target = batch_tensorify(batch_examples)
        batch_features, batch_target = batch_features.double(), batch_target.double()

        if gpu_enabled:
            input_batch = input_batch.cuda()
            labels_batch = labels_batch.cuda()

        optimizer.zero_grad()

        batch_features.to(device)
        batch_target.to(device)

        output, hidden = model(batch_features.double())

        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}/{param_epochs} ............. Loss: {loss.item()}')

torch.save(model.state_dict(), f'./{INSTRUMENT}_{param_epochs}epochs_model_state.pt')
