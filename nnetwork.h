//
// Created by Krzysiek on 08.06.2023.
//

#ifndef NEURALNETWORK_NNETWORK_H
#define NEURALNETWORK_NNETWORK_H
#include <stddef.h>
#include "matrix.h"

#define PRINT_NN(nn) nn_print((nn),(#nn))

#define PRINT_TD(td) td_print((td),(#td))

#define NN_INPUT(nn) (nn).activations[0]

#define NN_OUTPUT(nn) (nn).activations[(nn).count]

typedef struct{
    size_t count; // layers-1 because output doesnt have weeights and biases after itself
    Matrix *weights;
    Matrix *delta_w;
    Matrix *delta_b;
    Matrix *biases;
    Matrix *weighted_sum;
    Matrix *activations;
}Network;

typedef struct{
    size_t datasets;
    size_t in_count;
    size_t out_count;
    Matrix *input;
    Matrix *output;
}TData;

Network nn_allocate(size_t num_of_layers,size_t *architecture);

TData td_allocate(size_t in_count,size_t out_count,size_t datasets);

void free_td(TData training_d);

void pass_data(TData train);

void load_network(Network nn);

void nn_randomize(Network nn ,double min,double max);

void td_print(TData t_data , char* name);

void nn_print(Network nn , char* name);

void nn_clean(Network nnG);

void free_network(Network nn);

void save_values(Network nn);

#endif //NEURALNETWORK_NNETWORK_H
