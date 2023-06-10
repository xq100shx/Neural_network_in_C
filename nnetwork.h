//
// Created by Krzysiek on 08.06.2023.
//

#ifndef NEURALNETWORK_NNETWORK_H
#define NEURALNETWORK_NNETWORK_H
#include <stddef.h>
#include "matrix.h"
#define PRINT_NN(nn) nn_print((nn),(#nn))
#define NN_INPUT(nn) (nn).activations[0]
#define NN_OUTPUT(nn) (nn).activations[(nn).count]
typedef struct{
    size_t count; // layers-1 because output doesnt have weeights and biases after itself
    Matrix *weights;
    Matrix *biases;
    Matrix *activations;
}Network;
Network nn_allocate(size_t num_of_layers,size_t *architecture);
void nn_randomize(Network nn ,float l_range,float h_range);
void nn_print(Network nn , char* name);
void nn_clean(Network nnG);
void free_network(Network nn);
void save_values(Network nn);
#endif //NEURALNETWORK_NNETWORK_H
