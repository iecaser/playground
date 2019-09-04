import json
from loguru import logger
import pandas as pd


def decode_json_like(filein, fileout=None, selected_features=None):
    df = {}
    with open(filein)as f:
        for line in f:
            data = json.loads(line.strip())
            for key, value in data.items():
                df.setdefault(key, []).append(value)
    df = pd.DataFrame(df)
    if fileout is not None:
        df.to_csv(fileout, sep=',', index=False)
        logger.info(f'Saved output -> {fileout}.')
    return df if selected_features is None else df[selected_features]


filein = 'x.txt'
# test1
df1 = decode_json_like(filein)
logger.info('TEST1')
print(df1)

# test2
selected_features = ['source_ip', 'destination_ip']
df2 = decode_json_like(filein, fileout='y.txt',
                       selected_features=selected_features)
logger.info('TEST2')
print(df2)
