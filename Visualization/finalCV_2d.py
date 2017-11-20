import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np

def main():
    fourSame_in = open("finalCV/finalCV_fourSame.csv", 'r')
    fourDec_in = open("finalCV/finalCV_fourDec.csv", 'r')
    oneLayer_in = open("finalCV/finalCV_oneLayer.csv", 'r')

    fourSame_df = pandas.read_csv(fourSame_in)
    fourDec_df = pandas.read_csv(fourDec_in)
    oneLayer_df = pandas.read_csv(oneLayer_in)

    #fourSame x and y
    fourSame_ratios = fourSame_df.get("Ratio")
    fourSame_acc = fourSame_df.get("test_accuracy")

    #fourDec x and y
    fourDec_ratios = fourDec_df.get("Ratio")
    fourDec_acc = fourDec_df.get("test_accuracy")

    #oneLayer x and y
    oneLayer_ratios = oneLayer_df.get("Ratio")
    oneLayer_acc = oneLayer_df.get("test_accuracy")

    fourSame = plt.plot(fourSame_ratios, fourSame_acc, label='Four Same Size')
    fourDec = plt.plot(fourDec_ratios, fourDec_acc, label='Four Decreasing Size')
    oneLayer = plt.plot(oneLayer_ratios, oneLayer_acc, label='One Layer')

    #plot formatting
    plt.legend(loc='upper left', fontsize = 16)
    plt.ylim([.9,1.0])
    plt.title('Training Ratio and Accuracy', fontsize = 28)
    plt.xlabel('Training Ratio', fontsize = 22)
    plt.ylabel('Accuracy', fontsize = 22)
    plt.tick_params(labelsize=16)
    plt.grid()

    plt.show()

if __name__=="__main__":
    main()
