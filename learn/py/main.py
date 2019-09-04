import json
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd


def decode_json_like(filein, fileout=None, selected_features=None) -> pd.DataFrame:
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


def gen_graph_from_df(df, src, dst):
    def gen_graph(srcs, dsts):
        graph = nx.DiGraph()
        graph.add_edges_from(zip(srcs, dsts))
        return graph
    df = df[df[src] != df[dst]]
    df = df.drop_duplicates(subset=[src, dst])
    src_cnt, dst_cnt = df[src].unique().shape[0], df[dst].unique().shape[0]
    logger.info(
        f'Unique source ip: {src_cnt}, unique destination ip: {dst_cnt}.')
    graph = gen_graph(df[src], df[dst])
    return graph


filein = 'x.txt'
fileout = 'y.txt'
# test1
logger.info('TEST1')
df = decode_json_like(filein, fileout)
graph = gen_graph_from_df(df, src='source_ip', dst='destination_ip')

# same with you
pos = nx.spring_layout(graph)
nx.draw_networkx(graph, arrows=True, with_labels=True, pos=pos)
plt.savefig('graph.png')
