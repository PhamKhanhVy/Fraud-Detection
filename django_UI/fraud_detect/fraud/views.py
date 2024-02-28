import dgl
import pandas as pd
import numpy as np
import os
import json
import time
import random
import torch as th
from dgl.dataloading import DataLoader
from django.shortcuts import render
from .plot_graph import plot_neighborhood, neighborhood_stats, display_live_plot
from .utils import convert_timestamp, get_corresponding_mask, infer, node_construct
from .GNNmodels.model import RGCN
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql import Row
from datetime import datetime
from random import randint
from IPython.display import clear_output
from pyspark.sql.functions import from_json
import plotly.express as px
from kafka import KafkaConsumer
from time import sleep
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import JsonResponse
np.bool = np.bool_
import asyncio

scala_version = '2.12'
spark_version = '3.5.0'
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.6.0' #your kafka version
]
spark = SparkSession.builder.master("local").appName("kafka-example").config("spark.jars.packages", ",".join(packages)).getOrCreate()

def index(request):
    return render(request, 'index.html',)

def plot_graph(request):
    try:
      n_hops = int(request.GET.get('n_hops'))
      N_plots = int(request.GET.get('N_plots'))
      is_Fraud = int(request.GET.get('Is_Fraud'))
      batch_size = int(request.GET.get('batch_size'))

      loaded_graphs, _ = dgl.load_graphs('fraud/data/heterogeneous_graph.bin')
      hg = loaded_graphs[0]
      sampler = dgl.dataloading.MultiLayerNeighborSampler([10]*n_hops)

      df = pd.read_csv('fraud/data/data_preprocessed.csv')
      y_txn = df.sort_values(by='txnIdx')['FraudResult'].values
      save_dir = "fraud/static/neighborhood_plots"
      files = os.listdir(save_dir)
      if files:
          for file in files:
              file_path = os.path.join(save_dir, file)
              os.remove(file_path)
      dataloader_legit = DataLoader(
          hg, {'txnIdx': np.where(y_txn == is_Fraud)[0]}, sampler,
          batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
      plot_neighborhood(dataloader_legit, save_dir, N_plots=N_plots)
      plot_images = os.listdir(save_dir)
      return render(request, 'plot.html', {'plot_images': plot_images})
    except:
        return render(request, 'plot.html')

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
)

schema = StructType([
    StructField("TransactionId", StringType(), True),
    StructField("BatchId", StringType(), True),
    StructField("AccountId", StringType(), True),
    StructField("SubscriptionId", StringType(), True),
    StructField("CustomerId", StringType(), True),
    StructField("CurrencyCode", StringType(), True),
    StructField("CountryCode", IntegerType(), True),
    StructField("ProviderId", StringType(), True),
    StructField("ProductId", StringType(), True),
    StructField("ProductCategory", StringType(), True),
    StructField("ChannelId", StringType(), True),
    StructField("Amount", DoubleType(), True),
    StructField("Value", IntegerType(), True),
    StructField("TransactionStartTime", TimestampType(), True),
    StructField("PricingStrategy", IntegerType(), True)
])

kafka_params = {
    "kafka.bootstrap.servers": "localhost:9092",
    "subscribe": "test_streaming",
    "startingOffsets": "earliest"
}


def kafka_producer(request):
    if request.method == 'POST':
        sent_messages = []
        topic = 'test_streaming'
        df_test = pd.read_csv('fraud/data/test.csv')
        batch_size = 5000
        total_rows = len(df_test)
        random_indices = random.sample(range(total_rows), batch_size)
        random_rows = df_test.iloc[random_indices]

        for index, row in random_rows.iterrows():
            row_dict = row.to_dict()
            row_dict = {key: convert_timestamp(value) if isinstance(value, datetime) else value for key, value in row_dict.items()}

            producer.send(topic, value=row_dict)
            sent_messages.append(row_dict)
        producer.flush()
        return render(request, 'fraud_detect.html', {'sent_messages': sent_messages})
    else:
        return render(request, 'fraud_detect.html')


def live_view(request):
    loaded_graphs, _ = dgl.load_graphs('fraud/data/heterogeneous_graph.bin')
    hg = loaded_graphs[0]
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    df = pd.read_csv('fraud/data/data_preprocessed.csv')
    categorical = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    node_feat = node_construct(df, categorical)
    model_kwargs = dict(
        in_feats=node_feat.shape[1], h_feats=256, num_classes=2,
        num_layers=2, dropout=0.1, num_bases=None,
        self_loop=False, bn=True,
    )
    model = RGCN(hg, **model_kwargs).to(device)
    live_graph_html = ""
    kafka_stream_df = (
        spark.read.format("kafka")
        .option("kafka.bootstrap.servers", kafka_params["kafka.bootstrap.servers"])
        .option("subscribe", kafka_params["subscribe"])
        .option("startingOffsets", kafka_params["startingOffsets"])
        .load()
        .selectExpr("CAST(value AS STRING)")
        .select(from_json("value", schema).alias("data"))
        .select("data.*")
    )
    kafka_stream_df = kafka_stream_df.dropDuplicates().toPandas()
    mask = get_corresponding_mask(kafka_stream_df, df)
    tensor_mask = th.tensor(mask)
    y_pred = infer(model.to(device), hg.to(device), tensor_mask.to(device) , 'fraud/GNNmodels/rgcn.pt')
    y_pred_labels = (y_pred[:, 1] > 0.21062215).int().numpy()
    kafka_stream_df["Fraud_Predict"] = y_pred_labels
    live_graph_html = display_live_plot(kafka_stream_df)
    temp = kafka_stream_df[['TransactionId', 'Fraud_Predict']].to_html(index=False)
    unique_count = kafka_stream_df['TransactionId'].nunique()
    count_fraud_predicted = (kafka_stream_df["Fraud_Predict"] == 1).sum()
    context = {
        'live_graph_html': live_graph_html,
        'result' : temp,
        'unique_count' : unique_count,
        'fraud': count_fraud_predicted,
    }
    return render(request, 'live_view.html', context=context)
