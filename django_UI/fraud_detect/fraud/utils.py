import torch as th
import dgl
from copy import deepcopy
from dgl.dataloading import DataLoader
import numpy as np
import pandas as pd
def convert_timestamp(ts):
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def predict_from_probabilities(y_probs):
    predicted_classes = th.argmax(y_probs, dim=1)
    return predicted_classes

def get_corresponding_mask(df, df_transformed):
    df_unique_ids = df["TransactionId"].drop_duplicates()
    transaction_id_list = df_unique_ids.tolist()

    df_final = df_transformed.copy()
    df_final["mask"] = df_final["TransactionId"].isin(transaction_id_list)

    mask = df_final["mask"].values.astype(bool)
    return mask

@th.no_grad()
def infer(model_cp, hg, mask, best_model_fp=None):
    if best_model_fp:
        print("loading from disk")
        model = deepcopy(model_cp)
        model.load_state_dict(th.load(best_model_fp))
    else:
        model = model_cp
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    features = hg.ndata['feat']
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15]*len(model.convs))
    dataloader = DataLoader(
        hg, {'txnIdx': th.where(mask)[0]}, sampler,
        batch_size=1024, shuffle=False, drop_last=False, num_workers=0)

    y_preds = list()
    for input_nodes, output_nodes, blocks in dataloader:
        h = {k: features[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
        blocks = [b.to(device) for b in blocks]

        logits = model(blocks, h)
        y_preds.append(logits.softmax(dim=1))

    return th.cat(y_preds).cpu()

def node_construct(df, categorical):
    #One-Hot Encoding for Categorical Features:
    X_ohe = pd.get_dummies(df[categorical].astype(str), drop_first=True).values
    #Binary Encoding for 'Amount' Feature:
    X_amt = ((df['Amount']>0)*1).values
    #Logarithmic Transformation for 'Value' Feature:
    X_value = df['Value'].apply(np.log10).values
    #concatenation of Numerical Features:
    X_num = np.concatenate([X_amt.reshape((-1,1)), X_value.reshape((-1,1))], axis=1)
    #Min-Max Scaling:
    X_num = (X_num - X_num.mean(axis=0))/(X_num.max(axis=0) - X_num.min(axis=0))
    node_feat = np.concatenate([X_ohe, X_num], axis=1)
    return node_feat
