import networkx as nx
import dgl
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from dgl import function as fn
import torch as th
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def draw_kkl_hg(hg, **kwargs):

    hg_label = hg.ndata['label']['txnIdx'].int()

    g = dgl.to_homogeneous(hg)
    g.ndata['ntype'] = g.ndata['_TYPE']
    nx_G = g.to_networkx(node_attrs=['ntype']).to_undirected()
    fig, ax = plt.subplots(figsize=(10,10))

    pos = nx.spring_layout(nx_G, k=5/np.sqrt(g.num_nodes()))

    nodeShapes = ["^", "s", "o", "v"]
    # For each node class...
    for idx, ntype in enumerate(hg.ntypes):
        aShape = nodeShapes[idx]
        if ntype=='txnIdx':
            node_color = ['blue' if v==0 else 'red' if v==1 else 'black' for v in hg_label]
        else:
            node_color = "grey"
        nx.draw_networkx_nodes(
            nx_G, pos, node_shape=aShape, node_color=node_color, cmap='bwr', node_size=200,
            nodelist = [sNode[0] for sNode in nx_G.nodes(data=True) if sNode[1]["ntype"] == idx]
        )

    # Draw the edges between the nodes
    nx.draw_networkx_edges(nx_G, pos)

def plot_neighborhood(dataloader, save_dir, N_plots=5):
    loaded_graphs, _ = dgl.load_graphs('fraud/data/heterogeneous_graph.bin')
    hg = loaded_graphs[0]
    nstats = dict()
    for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        if i >= N_plots:
            break
        hg_tmp = dgl.node_subgraph(hg, input_nodes)

        # Generate a unique filename for each plot image
        filename = os.path.join(save_dir, f'neighborhood_plot_{i}.png')

        # Create the plot and save it to the specified directory
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_kkl_hg(hg_tmp)
        plt.savefig(filename)
        plt.close(fig)

def neighborhood_stats(hg):
    def mp_fn_dict(in_feat, out_feat):
        return {
            cetype: (fn.copy_u(in_feat, 'm'), fn.sum('m', out_feat))
            for cetype in hg.canonical_etypes
        }

    with hg.local_scope():
        degree_data = dict()
        for ntype in hg.ntypes:
            in_degrees = th.zeros(hg.num_nodes(ntype))
            for cetype in hg.canonical_etypes:
                if cetype[2]==ntype:
                    in_degrees += hg.in_degrees(etype=cetype)
            degree_data[ntype] = in_degrees

        hg.ndata['degree'] = degree_data
        hg.ndata['N_fraud'] = {
            ntype: (labels==1).float() for ntype, labels in hg.ndata['label'].items()
        }

        hg.multi_update_all(mp_fn_dict('degree', 'd_sum'), cross_reducer='sum')
        hg.multi_update_all(mp_fn_dict('N_fraud', 'total_fraud'), cross_reducer='sum')
        hg.multi_update_all(mp_fn_dict('total_fraud', 'total_fraud_2hop'), cross_reducer='sum')

        n_2hop = hg.ndata['d_sum']['txnIdx']
        fraud_rate_2hop = hg.ndata['total_fraud_2hop']['txnIdx'] / hg.ndata['d_sum']['txnIdx']
    return n_2hop, fraud_rate_2hop

def display_live_plot(df):
    fraud_df = df[df['Fraud_Predict'] == 1]
    fig = px.scatter(fraud_df, x="TransactionStartTime", y="Amount", color="Fraud_Predict", title="Live Streaming Data")
    line_fig = px.line(df, x="TransactionStartTime", y="Amount", title="Live Streaming Data")

    for trace in line_fig.data:
        fig.add_trace(trace)\

    fig.update_layout(
        autosize=False,
        width=1300,
        height=600,
    )
    return fig.to_html()
