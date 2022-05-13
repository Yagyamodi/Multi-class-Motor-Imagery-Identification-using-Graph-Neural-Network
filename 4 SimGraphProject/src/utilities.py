import glob
import numpy as np
from tqdm import tqdm, trange
import argparse
import json
import math
from myParser import parameter_parser
import random
import tensorflow as tf
import networkx as nx
from hausdorff import hausdorff_distance


def find_loss(prediction, target):
    prediction = prediction
    target = target
    score = (prediction-target)**2
    return score


def process(path):
    data = json.load(open(path))
    return data

class data2:

    def __init__(self, parser):
        self.args = parser
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        self.graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for self.graph_pair in tqdm(self.graph_pairs):
            self.data = process(self.graph_pair)
            self.global_labels = self.global_labels.union(set(self.data["labels_1"]))
            self.global_labels = self.global_labels.union(set(self.data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
        #print("Number of labels:")
        #print(self.number_of_labels)
        #print(self.global_labels)
    
    def getlabels(self):
        return self.global_labels
    def getnumlabels(self):
        return self.number_of_labels
    def gettrain(self):
        return self.training_graphs
    def gettest(self):
        return self.testing_graphs
    def create_batches(self):
        #random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), 1):
            batches.append(self.training_graphs[graph:graph+1])
        return batches

def function(x):
    #weight = int((x+0.025)/0.05)
    weight = int(round( round(x,2)/0.15 , 0))*1.0
    # return round(weight/3, 2);

    return weight/2;

def find_hausdorff_distance(edges_1, edges_2):
    dist = []
    for i in range(0,9):
        dist.append(hausdorff_distance(np.array([edges_1[i]]), np.array([edges_2[i]]), distance='manhattan'))
        
    dist = np.array(dist)
    return dist
    

def convert_to_keras(data, global_labels):
        transformed_data = dict()
        
        edges_1 = np.array(data["adj_matrix_1"])
        edges_2 = np.array(data["adj_matrix_2"])
        func = lambda x : 1.0 if (round(x,2)>0.15) else 0.0
        for i in range(0,9):
            for j in range(0,9):  
                edges_1[i][j] = func(edges_1[i][j])
                edges_2[i][j] = func(edges_2[i][j])
        #print("edges_2:  ", edges_2)

        """ 
        Feature transforming 
        """
        features_1, features_2 = [], []
        for n in data["labels_1"]:
            features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        for n in data["labels_2"]:
            features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

        #features_1 = np.around(np.array(data["features_1"]), decimals=3)
        #features_1 = np.multiply(np.array(data["features_1"]), 1)
        #features_2 = np.multiply(np.array(data["features_2"]), 1)


        features_1 = tf.convert_to_tensor(features_1, dtype=tf.float32)
        features_2 = tf.convert_to_tensor(features_2, dtype=tf.float32)
        transformed_data["edge_index_1"] = edges_1
        transformed_data["edge_index_2"] = edges_2
        transformed_data["features_1"] = features_1
        transformed_data["features_2"] = features_2
        #norm_ged = find_hausdorff_distance(edges_1, edges_2) 

        norm_ged = hausdorff_distance(edges_1, edges_2, distance='manhattan')
        
        #calculate_ged(adj_matrix_1, adj_matrix_2)/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        #norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        
        transformed_data["target"] = tf.reshape(tf.convert_to_tensor(np.exp(-norm_ged).reshape(1, 1)),-1)
        #print(transformed_data["target"].shape)
        return transformed_data    

    
#x = data2()