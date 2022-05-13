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
import matplotlib.pyplot as plt

manual_variable_initialization(True)

parser = parameter_parser()
my_logs = ""


def train(model, x):
    batches = x.create_batches()
    global_labels = x.getlabels()
    """
    Training the Network
    Take every graph pair and train it as a batch.
    """
    t_x = x
    last=0
    #for epoch in range(0,parser.epochs):
    epoch = 0;
    p = [100];
    all_p = []
    while (epoch<parser.epochs or float(p[0])>0.05):
        p = 0
        
        print("EPOCH NUMBER: ", epoch)
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for graph_pair in batch:
                data = process(graph_pair)
                data = convert_to_keras(data, global_labels)
                x, y, a, b = [ np.array([ data["features_1"] ]), np.array([ data["features_2"] ]), np.array([ data["edge_index_1"] ]), np.array([ data["edge_index_2"] ]) ]
                p = model.train_on_batch([x, a, y, b], data["target"], reset_metrics = False)
        if epoch%(parser.saveafter) == 0:
                print("Train Error:")
                print(p[0]*100)
                #model.save("train")
                #model.save_weights("xweights")
                
        epoch = epoch+1
        all_p.append(p[0]*100)

    dot_img_file = "C:/Users/91876/Desktop/Signal Processing/Results plot/trainplot.png"
    plt.plot(all_p)
    plt.xlabel('Epochs')
    plt.ylabel('Train error')
     
    plt.title('A01T_4')
    
    plt.show()
    #keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

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
    print_str = "Model test error: " +str(round(model_error, 5)*100)+".\n"
    print(print_str)
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
    
    """print("Printing model summary: ")
    model.summary()
    model.save("train")
    
    x : Data loading
    train used to train
    test over the test data
    """
    x = data2(parser)
    model = train(model, x)
    model_error = test(model, x)
    return model_error


if __name__ == "__main__":
    
    for train_file_no in range(4,5):
        train_file_name = "C:/Users/91876/Downloads/SimGNN-main/dataset/A01/Training/A01T_" + str(train_file_no) + "/"
        my_logs += "\nTraining data is " + "A01T_" + str(train_file_no) + ".\n\nTesting data is -"
        for test_file_no in range(4,5):
            test_file_name = "C:/Users/91876/Downloads/SimGNN-main/dataset/A01/Testing/A01E_" + str(test_file_no) + "/"
            parser.training_graphs = train_file_name
            parser.testing_graphs = test_file_name
            print("Training data: ", parser.training_graphs)
            print("Testing data: ", parser.testing_graphs)
            model_error = main()
            print_str = "Model test error: " +str(round(model_error, 5)*100)+".\n"
            my_logs += "A01E_" + str(test_file_no) + ": " + print_str
            print("----------------------------")
            
    print("LOGS: ", my_logs)
    

