//
// Created by Krzysiek on 08.06.2023.
//

#ifndef NEURALNETWORK_MLEARNING_H
#define NEURALNETWORK_MLEARNING_H
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include "mlearning.h"
#include "nnetwork.h"
#include "matrix.h"
#include "actvfunc.h"

void forward(Network nn);

double cost(Network nn,TData train_data);

size_t backpropagation(Network nn, Network nnG, TData train_data);

void learn(Network nn, Network nnG, double learning_rate);

double success(Network nn,TData train);

#endif //NEURALNETWORK_MLEARNING_H
