import sys
from tensorflow import keras
from keras import backend
from keras import applications
from keras import datasets
from keras import wrappers
from keras_gcn import GraphConv
import numpy as np
from tqdm import tqdm, trange
from myParser import parameter_parser
from utilities import data2, convert_to_keras, process, find_loss
from simgnn import simgnn
from custom_layers import Attention, NeuralTensorLayer
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

parser = parameter_parser()
print("parser " )
print(parser)

def train(model, x):
    batches = x.create_batches()
    global_labels = x.getlabels()

    t_x = x
    last=0
    for epoch in range(0,parser.epochs):
        p=0
        
        print("EPOCH NUMBER: ", epoch)
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for graph_pair in batch:
                data = process(graph_pair)
                data = convert_to_keras(data, global_labels)
                x, y, a, b = [ np.array([ data["features_1"] ]), np.array([ data["features_2"] ]), np.array([ data["edge_index_1"] ]), np.array([ data["edge_index_2"] ]) ]
                p = model.train_on_batch([x, a, y, b], data["target"], reset_metrics = False)
        if epoch%(parser.saveafter) == 0:
                print("Train Error:")
                print(p)
                model.save("train")
                model.save_weights("xweights")
    return model

def test(model, x):
    global_labels = x.getlabels()
    test = x.gettest()
    scores = []
    g_truth = []
    for graph_pair in tqdm(test):
        data = process(graph_pair)
        data = convert_to_keras(data, global_labels)
        x = np.array([ data["features_1"] ])
        y = np.array([ data["features_2"] ])
        a = np.array([ data["edge_index_1"] ])
        b = np.array([ data["edge_index_2"] ])
        g_truth.append(data["target"])
        y=model.predict([x, a, y, b])
        scores.append(find_loss(y, data["target"]))

    norm_ged_mean = np.mean(g_truth)
    model_error = np.mean(scores)
    print("\nModel test error: " +str(round(model_error, 5))+".")
    return model_error

def main():
    model = simgnn(parser);
    opt = keras.optimizers.Adadelta(learning_rate=parser.learning_rate, rho=parser.weight_decay)
    #opt = keras.optimizers.Adam(learning_rate=parser.learning_rate)
    model.compile(
                optimizer=opt,
                loss='mse',
                metrics=[keras.metrics.MeanSquaredError()],
            )
    
    print("Printing model summary: ")
    model.summary()
    model.save("train")

    x = data2()
    model = train(model, x)
    test(model, x)


if __name__ == "__main__":
    main()