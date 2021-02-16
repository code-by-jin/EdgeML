import matplotlib.pyplot as plt
from utils import load_pickle

def plot_loss_acc(pickle_file):
    data = load_pickle(pickle_file)
    