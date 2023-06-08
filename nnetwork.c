//
// Created by Krzysiek on 08.06.2023.
//
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "matrix.h"
#include "nnetwork.h"

Network nn_allocate(size_t num_of_layers,size_t *architecture){
    assert(num_of_layers > 1);
    Network nn;
    nn.count = num_of_layers-1;
    nn.activations = calloc(nn.count+1,sizeof(*nn.activations));
    assert(nn.activations != NULL);
    nn.weights = calloc(nn.count,sizeof(*nn.weights));
    assert(nn.weights != NULL);
    nn.biases = calloc(nn.count,sizeof(*nn.biases));
    assert(nn.biases!= NULL);
    for(int i=0;i<nn.count;i++){
        nn.activations[i] = matrix_allocate(architecture[i],1);
        nn.weights[i] = matrix_allocate(nn.activations[i].cols,architecture[i+1]);
        nn.biases[i] = matrix_allocate(architecture[i+1],1);
    }
    nn.activations[nn.count] = matrix_allocate(architecture[nn.count],1);
    return nn;
}
void nn_randomize(Network nn ,float l_range,float h_range);
void nn_print(Network nn , char* name){
    size_t i=0,j=0;
    char buf[256];
    printf("%s = [\n",name);
    for(i = 0;i<nn.count;i++){
        snprintf(buf,sizeof(buf),"Activation %zu",i);
        matrix_print(nn.activations[i],buf);
        snprintf(buf,sizeof(buf),"Weights %zu",i);
        matrix_print(nn.weights[i],buf);
        snprintf(buf,sizeof(buf),"Biases %zu",i);
        matrix_print(nn.biases[i],buf);
    }
    snprintf(buf,sizeof(buf),"Weighted sum");
    snprintf(buf,sizeof(buf),"Output");
    matrix_print(NN_OUTPUT(nn),buf);
    printf("]\n");
}
void nn_clean(Network nnG);
void free_network(Network nn);
void save_values(Network nn);
//[2,2,1] num of layers 3