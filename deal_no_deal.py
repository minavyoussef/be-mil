#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DATA_DIR = './data'
RESOLUTION = 'daily'
REGION = 'us'
INSTRUMENT = './nyse stocks/2/unh.us.txt'

ohlc = ['open', 'high', 'low', 'close']
target_col = 'close'

data_file = os.path.normpath(os.path.join(DATA_DIR, RESOLUTION, REGION, INSTRUMENT))

df = pd.read_csv(data_file)
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'openint']

df.date = pd.to_datetime(df.date, format='%Y%m%d')

df['high_open_delta'] = df['high'] - df['open']
df['low_close_delta'] = df['low'] - df['close']

df['high_low_delta'] = df['high'] - df['low']
df['open_close_delta'] = df['open'] - df['close']

df['high_close_delta'] = df['high'] - df['close']
df['low_open_delta'] = df['low'] - df['open']

df[f'{target_col}_t1'] = df[target_col].shift(-1)
df[f'{target_col}_delta'] = df[f'{target_col}_t1'] - df[target_col]

df[['volume']] = StandardScaler().fit_transform(df[['volume']])

df.drop([f'{target_col}_t1'], axis=1, inplace=True)
df.drop(ohlc + ['date', 'openint'], axis=1, inplace=True)
df = df[:-1]

features = ['volume',
            'high_open_delta',
            'low_close_delta',
            'high_low_delta',
            'open_close_delta',
            'high_close_delta',
            'low_open_delta']
torch_tensor = torch.tensor(df[features].values)


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


train_dataset = train_dataset_generator(df, shift_range=5, repeat_out=1000)


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


train_dataset = pad_dataset_sequence(train_dataset)
for e in train_dataset:
    print(len(e))