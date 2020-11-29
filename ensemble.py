# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os

ensemble_weights = {
    'MODEL1/submission.csv': 1,
    'MODEL2/submission.csv': 1,
    'MODEL3/submission.csv': 1,
    'MODEL4/submission.csv': 1,
}

ensembled = None

weight_sum = 0.
for file_path in ensemble_weights:
    print('Reading {}...'.format(file_path))
    weight = ensemble_weights[file_path]
    weight_sum += weight
    df = pd.read_csv(file_path)
    if ensembled is None:
        ensembled = df.copy()
        ensembled['value'] = 0.0
    ensembled['value'] += df['value'] * weight

ensembled['value'] = ensembled['value'] / weight_sum

config_path = "output/ensemble_config.json"
with open(config_path, "w") as json_file:
    print("Saving a ensemble config file:", config_path)
    json.dump(ensemble_weights, json_file)

submission_path = 'output/ensemble_submission.csv'
print('Saving a ensemble submission file:', submission_path)
ensembled.to_csv(submission_path, index=False)