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
double cost(Network nn,size_t input_size,size_t output_size,size_t data_sets);
size_t backpropagation(Network nn, Network nnG, size_t input_size,size_t output_size,size_t data_sets);
void learn(Network nn, Network nnG, double learning_rate);
bool is_valid(float x);
#endif //NEURALNETWORK_MLEARNING_H
