import json
from loguru import logger
import pandas as pd


def decode_json_like(filein, fileout=None, selected_features=None):
    df = {}
    with open(filein)as f:
        for line in f:
            data = json.loads(line.strip())
            for key, value in data.items():
                if selected_features is not None and key not in selected_features:
                    continue
                df.setdefault(key, []).append(value)
    df = pd.DataFrame(df)
    if fileout is not None:
        df.to_csv(fileout, sep=',', index=False)
        logger.info(f'Saved output -> {fileout}.')
    return df


filein = 'x.txt'
# test1
df1 = decode_json_like(filein)
logger.info('TEST1')
print(df1.head())

# test2
selected_features = ['source_ip', 'destination_ip']
df2 = decode_json_like(filein, fileout='y.txt',
                       selected_features=selected_features)
logger.info('TEST2')
print(df2.head())
