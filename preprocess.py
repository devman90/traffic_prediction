# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

print('Reading csv files...')
all_data = pd.concat([
    pd.read_csv('input/training_data.csv'),
    pd.read_csv('input/test_data.csv')
])

# 요일 이상함 -> 날짜를 적절히 옮긴다.
import datetime
from dateutil.parser import parse
def to_datetime(s):
    dt = parse(s)
    if parse('2008-11-19 00:00:00') > dt >= parse('2008-11-18 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2008-11-12 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2008-10-20 00:00:00'):
        dt += datetime.timedelta(days=1)
    if dt >= parse('2008-04-01 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2008-12-07 00:00:00'):
        dt += datetime.timedelta(days=1)
    if dt >= parse('2008-12-26 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2009-01-01 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2009-01-20 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2009-02-14 00:00:00'):
        dt -= datetime.timedelta(days=1)
    if dt >= parse('2009-02-23 00:00:00'):
        dt += datetime.timedelta(days=1)
    if dt >= parse('2009-03-06 00:00:00'):
        dt -= datetime.timedelta(days=1)
    return dt
    
print('Preprocessing datetimes...')
all_data['timestamp'] = all_data['timestamp'].apply(to_datetime)

all_data = all_data.reset_index()
del all_data['index']

data_path = 'preprocessed/preprocessed_data.csv'
all_data.to_csv(data_path, index=False)
print('Saved:', data_path)

# Generating meta dataset
print('Preprocessing meta data...')
stations = all_data.mean(numeric_only=True).index
means = all_data.mean(numeric_only=True).values
stds = all_data.std(numeric_only=True).values
meta_path = 'preprocessed/preprocessed_meta.csv'
pd.DataFrame({
    'ID': stations,
    'Mean': means,
    'Std': stds
}).to_csv(meta_path)
print('Saved:', meta_path)