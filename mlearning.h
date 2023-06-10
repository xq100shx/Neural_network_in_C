//
// Created by Krzysiek on 08.06.2023.
//

#ifndef NEURALNETWORK_MLEARNING_H
#define NEURALNETWORK_MLEARNING_H
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include "mlearning.h"
#include "nnetwork.h"
#include "matrix.h"
#include "actvfunc.h"
void forward(Network nn);
float cost(Network nn,size_t input_size,size_t output_size,size_t data_sets);
size_t backpropagation(Network nn, Network nnG, size_t input_size,size_t output_size,size_t data_sets);
void learn(Network nn, Network nnG, float learning_rate);
#endif //NEURALNETWORK_MLEARNING_H
