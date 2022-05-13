import argparse
import keras


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="C:/Users/yagya/Desktop/SimGraphProject/dataset/A01/Training/A01T_4/",
	                help="Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="C:/Users/yagya/Desktop/SimGraphProject/dataset/A01/Testing/A01E_4/",
	                help="Folder with testing graph pair jsons.")
            

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 5.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=256,
	                help="Filter (neurons) in 1st convolution. Default is 128.")


    parser.add_argument("--filters-2",
                        type=int,
                        default=128,
	                help="Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
	                help="Filters (neurons) in 3rd convolution. Default is 32.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.005,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--saveafter",
                        type=int,
                        default=5,
	                help="Saves model after every argument epochs")

    return parser.parse_args()
