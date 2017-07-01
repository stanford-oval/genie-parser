#!/usr/bin/python3

import numpy as np

import sys
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import json

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=20) 

def correct_function():
    # order is para-prim, para-comp, cheat-prim, cheat-comp, scenario-prim, scenario-comp
    SEMPRE = [85.04, 66.98, 77.5, 49.01, 60, 33]
    DEEP_SEMPRE = [95.23, 75.64, 50, 47.05, 42.85, 16.66]
    
    X = np.arange(3)
    width = (0.8-0.1)/4

    s_p = [SEMPRE[0], SEMPRE[2], SEMPRE[4]]
    s_c = [SEMPRE[1], SEMPRE[3], SEMPRE[5]]
    d_p = [DEEP_SEMPRE[0], DEEP_SEMPRE[2], DEEP_SEMPRE[4]]
    d_c = [DEEP_SEMPRE[1], DEEP_SEMPRE[3], DEEP_SEMPRE[5]]
    
    plt.bar(X, s_p, width=width, color='#85c1e5')
    plt.bar(X+width, d_p, width=width, color='#254e7b')
    plt.bar(X+2*width+0.1, s_c, width=width, color='#85c1e5')
    plt.bar(X+3*width+0.1, d_c, width=width, color='#254e7b')
    
    width = (0.8-0.1)/4
    plt.xticks(np.array([width, 3*width+0.1,
                         1+width, 1+3*width+0.1,
                         2+width, 2+3*width+0.1]),
        ["Prim.", "Comp.", "Prim.", "Comp.", "Prim.", "Comp."])
    plt.text(0.4, -10, "Paraphrasing", ha='center', fontsize=18)
    plt.text(1.4, -10, "Scenarios", ha='center', fontsize=18)
    plt.text(2.4, -10, "Composition", ha='center', fontsize=18)
    plt.ylim(0, 100)
    plt.xlim(-0.1, 2.9)
    #plt.tight_layout()
    plt.legend(["SEMPRE", "Neural Net"], loc ="upper right")
    plt.savefig('./figures/correct-function.pdf')

def accuracy_against_sempre():
    # order is para-prim, para-comp, cheat-prim, cheat-comp, scenario-prim, scenario-comp
    SEMPRE = [71.4, 50.2, 67.5, 33.3, 34.28, 30.5]
    DEEP_SEMPRE = [89.11, 55.27, 47.5, 29.4, 34.28, 16.66]
    
    X = np.arange(3)
    width = (0.8-0.1)/4

    s_p = [SEMPRE[0], SEMPRE[2], SEMPRE[4]]
    s_c = [SEMPRE[1], SEMPRE[3], SEMPRE[5]]
    d_p = [DEEP_SEMPRE[0], DEEP_SEMPRE[2], DEEP_SEMPRE[4]]
    d_c = [DEEP_SEMPRE[1], DEEP_SEMPRE[3], DEEP_SEMPRE[5]]
    
    plt.bar(X, s_p, width=width, color='#85c1e5')
    plt.bar(X+width, d_p, width=width, color='#254e7b')
    plt.bar(X+2*width+0.1, s_c, width=width, color='#85c1e5')
    plt.bar(X+3*width+0.1, d_c, width=width, color='#254e7b')
    
    width = (0.8-0.1)/4
    plt.xticks(np.array([width, 3*width+0.1,
                         1+width, 1+3*width+0.1,
                         2+width, 2+3*width+0.1]),
        ["Prim.", "Comp.", "Prim.", "Comp.", "Prim.", "Comp."])
    plt.text(0.4, -10, "Paraphrasing", ha='center', fontsize=18)
    plt.text(1.4, -10, "Scenarios", ha='center', fontsize=18)
    plt.text(2.4, -10, "Composition", ha='center', fontsize=18)
    plt.ylim(0, 100)
    plt.xlim(-0.1, 2.9)
    #plt.tight_layout()
    plt.legend(["SEMPRE", "Neural Net"], loc ="upper right")
    plt.savefig('./figures/accuracy-combined.pdf')
    
def extensibility():
    # order is new device acc, new device recall, new domain acc, new domain recall
    SEMPRE = [100 * 117./214., 100 * (10.+63.)/(15.+104.), 100 * (42.+232.)/(535.+75.), 100 * (32.+136.)/(286.+48.)]
    DEEP_SEMPRE = [38, 47, 55, 74]
    
    X = np.arange(2)
    width = (0.8-0.1)/4

    s_a = [SEMPRE[0], SEMPRE[2]]
    s_r = [SEMPRE[1], SEMPRE[3]]
    d_a = [DEEP_SEMPRE[0], DEEP_SEMPRE[2]]
    d_r = [DEEP_SEMPRE[1], DEEP_SEMPRE[3]]
    
    plt.bar(X, s_a, width=width, color='#85c1e5')
    plt.bar(X+width, d_a, width=width, color='#254e7b')
    plt.bar(X+2*width+0.1, s_r, width=width, color='#85c1e5')
    plt.bar(X+3*width+0.1, d_r, width=width, color='#254e7b')
    
    width = (0.8-0.1)/4
    plt.xticks(np.array([width, 3*width+0.1,
                         1+width, 1+3*width+0.1,
                         2+width, 2+3*width+0.1]),
        ["Accuracy", "Recall", "Accuracy", "Recall"])
    plt.text(0.4, -10, "New Device", ha='center', fontsize=18)
    plt.text(1.4, -10, "New Domain", ha='center', fontsize=18)
    plt.ylim(0, 100)
    plt.xlim(-0.1, 1.9)
    #plt.tight_layout()
    plt.legend(["SEMPRE", "Neural Net"], loc ="upper right")
    plt.savefig('./figures/extensibility.pdf')

def recall():
    # order is para-prim, para-comp, cheat-prim, cheat-comp, scenario-prim, scenario-comp
    SEMPRE = [81.06, 55.33, 65.38, 34.69, 40.0, 38.46]
    DEEP_SEMPRE = [93.75, 65.93, 60.0, 30.61, 58.33, 22.72]
    
    X = np.arange(3)
    width = (0.8-0.1)/4

    s_p = [SEMPRE[0], SEMPRE[2], SEMPRE[4]]
    s_c = [SEMPRE[1], SEMPRE[3], SEMPRE[5]]
    d_p = [DEEP_SEMPRE[0], DEEP_SEMPRE[2], DEEP_SEMPRE[4]]
    d_c = [DEEP_SEMPRE[1], DEEP_SEMPRE[3], DEEP_SEMPRE[5]]
    
    plt.bar(X, s_p, width=width, color='#85c1e5')
    plt.bar(X+width, d_p, width=width, color='#254e7b')
    plt.bar(X+2*width+0.1, s_c, width=width, color='#85c1e5')
    plt.bar(X+3*width+0.1, d_c, width=width, color='#254e7b')
    
    width = (0.8-0.1)/4
    plt.xticks(np.array([width, 3*width+0.1,
                         1+width, 1+3*width+0.1,
                         2+width, 2+3*width+0.1]),
        ["Prim.", "Comp.", "Prim.", "Comp.", "Prim.", "Comp."])
    plt.text(0.4, -10, "Paraphrasing", ha='center', fontsize=18)
    plt.text(1.4, -10, "Scenarios", ha='center', fontsize=18)
    plt.text(2.4, -10, "Composition", ha='center', fontsize=18)
    plt.ylim(0, 100)
    plt.xlim(-0.1, 2.9)
    #plt.tight_layout()
    plt.legend(["SEMPRE", "Neural Net"], loc ="upper right")
    plt.savefig('./figures/recall.pdf')

def different_training_sets():
    # base+author -> +paraphrasing -> +ifttt -> +generated
    train = [84.7, 93.2, 90.4, 91.99]
    test = [3.6, 37.4, 50.94, 55.4]
    train_recall = [66.6, 88.43, 92.63, 91.21]
    test_recall = [0.066, 49.05, 50.94, 75.47]
    
    #plt.newfigure()
    
    X = 1 + np.arange(4)
    plt.plot(X, train_recall, '--', color='#85c1e5')
    plt.plot(X, train, '-x', color='#6182a6')
    plt.plot(X, test_recall, '-o', color='#6182a6')
    plt.plot(X, test, '-', color='#052548')
    
    plt.ylim(0, 100)
    plt.xlim(0.5, 4.5)
    
    plt.xticks(X, ["Base + Author", "+ Paraphrasing", "+ IFTTT", "+ Generated"])
    plt.tight_layout()
    
    plt.legend(["Train recall", "Train accuracy", "Test recall", "Test accuracy"], loc='lower right')
    plt.savefig('./figures/training-sets.pdf')
    
def model_choices():
    # no attention: model 43
    # full: model 19
    # no grammar/full : model 19
    
    # no attention/no grammar, +grammar, +attention, full model
    train = [89.09, 89.16, 90.47, 90.49]
    dev = [45.7, 45.8, 55.5, 56.1] 
    test = [40.2, 40.4, 56, 56.6]
    train_recall = [82.30, 82.35, 90.04, 90.05]
    dev_recall = [62.62, 62.63, 76.76, 77.78]
    test_recall = [59.43, 60.37, 69.8, 70.75]
    
    #plt.newfigure()
    
    X = 1 + np.arange(4)
    plt.plot(X, train_recall, '--')#, color='#85c1e5')
    plt.plot(X, train, '--x')#, color='#6182a6')
    plt.plot(X, dev_recall, '-+')#
    plt.plot(X, dev, '-o')#
    plt.plot(X, test_recall, '-^')#, color='#6182a6')
    plt.plot(X, test, '-')#, color='#052548')
    
    plt.ylim(0, 100)
    plt.xlim(0.5, 4.5)
    
    plt.xticks(X, ["Seq2Seq", "+ Grammar", "+ Attention", "Full Model"])
    plt.tight_layout()
    
    plt.legend(["Train recall", "Train accuracy", "Dev recall", "Dev accuracy", "Test recall", "Test accuracy"], loc='lower right')
    plt.savefig('./figures/model-choices.pdf')

def dataset_train():
    # 0 param, 1 param, 2 param, 3+ param
    base = [1388, 1285, 977, 307]
    paraphrasing = [1185, 2277, 1471, 900]
    ifttt = [1525, 645, 414, 2607]
    generated = [569, 2098, 2723, 4610]
    
    data = np.array([base, paraphrasing, ifttt, generated])
    p_0 = data[:,0]
    p_1 = data[:,1]
    p_2 = data[:,2]
    p_3 = data[:,3]

    width = 0.7
    
    X = np.arange(4)
    plt.bar(X, p_3, width=width, color='#ffffff', bottom=p_0+p_1+p_2)
    plt.bar(X, p_2, width=width, color='#cde6f4', bottom=p_0+p_1)
    plt.bar(X, p_1, width=width, color='#85c1e5', bottom=p_0)
    plt.bar(X, p_0, width=width, color='#254e7b')
    
    plt.xticks(X + width/2, ["Base +\n Author", "Paraphrasing", "IFTTT", "Generated"])
    plt.xlim(-0.3, 4)
    plt.ylim(0, 11000)
    
    plt.tight_layout()
    plt.legend(["3+ Params", "2 Params", "1 Param", "0 Params"], loc='upper left')
    plt.savefig('./figures/dataset-train.pdf')

def dataset_test():
    # total, 0 param, 1 param, 2 param, 3+ param
    paraphrasing_prim = [73, 170, 44, 7]
    paraphrasing_comp = [103, 144, 103, 77]
    cheatsheet_prim = [32,8,0,0]
    cheatsheet_comp = [9, 18, 19, 5]
    scenario_prim = [22, 13, 0, 0]
    scenario_comp = [7, 19, 9, 1]

    prim_data = np.array([paraphrasing_prim, cheatsheet_prim, scenario_prim])
    comp_data = np.array([paraphrasing_comp, cheatsheet_comp, scenario_comp])
    
    p_0 = prim_data[:,0]
    p_1 = prim_data[:,1]
    p_2 = prim_data[:,2]
    p_3 = prim_data[:,3]
    c_0 = comp_data[:,0]
    c_1 = comp_data[:,1]
    c_2 = comp_data[:,2]
    c_3 = comp_data[:,3]
    
    X = np.arange(3)

    width = (0.8-0.1)/2
    plt.bar(X, p_3, width=width, color='#ffffff', bottom=p_0+p_1+p_2)
    plt.bar(X, p_2, width=width, color='#cde6f4', bottom=p_0+p_1)
    plt.bar(X, p_1, width=width, color='#85c1e5', bottom=p_0)
    plt.bar(X, p_0, width=width, color='#254e7b')
    plt.bar(X+0.1+width, c_3, width=width, color='#ffffff', bottom=c_0+c_1+c_2)
    plt.bar(X+0.1+width, c_2, width=width, color='#cde6f4', bottom=c_0+c_1)
    plt.bar(X+0.1+width, c_1, width=width, color='#85c1e5', bottom=c_0)
    plt.bar(X+0.1+width, c_0, width=width, color='#254e7b')
    
    plt.xticks(np.array([width/2, width+width/2+0.1,
                         1+width/2, 1+width+width/2+0.1,
                         2+width/2, 2+width+width/2+0.1]),
        ["Prim.", "Comp.", "Prim.", "Comp.", "Prim.", "Comp."])
    plt.text(0.4, -50, "Paraphrasing", ha='center', fontsize=20)
    plt.text(1.4, -50, "Scenarios", ha='center', fontsize=20)
    plt.text(2.4, -50, "Composition", ha='center', fontsize=20)
    plt.xlim(-0.1, 2.9)
    
    plt.legend(["3+ Params", "2 Params", "1 Param", "0 Params"])
    plt.savefig('./figures/dataset-test.pdf')

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def learning():
    with open('./data/train-stats.json', 'r') as fp:
        data = np.array(json.load(fp), dtype=np.float32)
    
    loss = data[:,0]
    train_acc = 100*data[:,1]
    dev_acc = 100*data[:,2]

    dev_mov_avg = movingaverage(dev_acc, 3)
    
    X = 1 + np.arange(len(data))
    plt.xlim(0, len(data)+1)
    
    #plt.plot(X, loss)
    #plt.ylabel('Loss')
    plt.xlabel('Training epoch', fontsize=20)
    
    #plt.gca().twinx()
    plt.plot(X, train_acc)
    plt.plot(X, dev_acc)
    plt.plot(X[1:-1], dev_mov_avg, '--')
    #plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")
    plt.savefig('./figures/learning.pdf')

def usage():
    print('Usage: ./gen_plot.py [learning]')

def do_all():
    raise RuntimeError("all is broken, don't use")
    accuracy_against_sempre()
    recall()
    correct_function()
    different_training_sets()
    learning()

def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    what = sys.argv[1]
    if what == 'all':
        do_all()
    elif what == 'learning':
        learning()
    elif what == 'accuracy-combined':
        accuracy_against_sempre()
    elif what == 'recall':
        recall()
    elif what == 'correct-function':
        correct_function()
    elif what == 'training-sets':
        different_training_sets()
    elif what == 'dataset-train':
        dataset_train()
    elif what == 'dataset-test':
        dataset_test()
    elif what == 'model-choices':
        model_choices()
    elif what == 'extensibility':
        extensibility()
    else:
        usage()
        sys.exit(1)
        
if __name__ == '__main__':
    main()
