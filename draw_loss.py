#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np


def draw_loss():
    epochs = np.arange(0, 150, 1)
    train_loss = []
  
    with open('train_loss.txt') as f:
        for line in f:
            train_loss.append(float(line.strip()))
    
    plt.plot( train_loss, label='train_loss')
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw_loss()

