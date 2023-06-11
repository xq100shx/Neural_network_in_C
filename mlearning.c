//
// Created by Krzysiek on 08.06.2023.
//
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include "mlearning.h"
#include "nnetwork.h"
#include "matrix.h"
#include "actvfunc.h"
bool is_valid(double x)
{
    return x*0.0==0.0;
}
void forward(Network nn){
    for(int i=0;i<nn.count-1;i++){
        matrix_dot_product(nn.weighted_sum[i],nn.weights[i],nn.activations[i]);
        matrix_sum(nn.weighted_sum[i],nn.biases[i]);
        matrixcpy(nn.activations[i+1],nn.weighted_sum[i]);
        ACTIVATE(nn.activations[i+1],ReLU); // keeping it as it is but would be nice to change it
                                               //Sigmoid/ReLU/LeakyReLU//
    }
    matrix_dot_product(nn.weighted_sum[nn.count-1],nn.weights[nn.count-1],nn.activations[nn.count-1]);
    matrix_sum(nn.weighted_sum[nn.count-1],nn.biases[nn.count-1]);
    matrixcpy(NN_OUTPUT(nn),nn.weighted_sum[nn.count-1]);
    softmax(NN_OUTPUT(nn));
}
double success(Network nn,TData train){
    size_t success = 0;

    for(int i=0;i<train.datasets;i++){
        matrixcpy(NN_INPUT(nn),train.input[i]);
        forward(nn);
        if(max_value_index_vector(NN_OUTPUT(nn))== max_value_index_vector(train.output[i])) success++;
    }
    return (double)success/(double) train.datasets;
}
double cost(Network nn,TData train_data){
    assert(NN_INPUT(nn).rows == train_data.in_count);
    assert(NN_OUTPUT(nn).rows == train_data.out_count);
    double cost = 0.0f;
    for(int n=0;n<train_data.datasets;n++) {
        matrixcpy(NN_INPUT(nn),train_data.input[n]);
        forward(nn);
        for(int i=0;i<train_data.out_count;i++){
            double difference = NN_OUTPUT(nn).ptr[i][0] - train_data.output[n].ptr[i][0];
            cost += difference * difference;
        }
    }
    return cost/(double)train_data.datasets;
}
size_t backpropagation(Network nn, Network nnG, TData train_data){
    assert(NN_INPUT(nn).rows == train_data.in_count);
    assert(NN_OUTPUT(nn).rows == train_data.out_count);
    nn_clean(nnG);
    for(int k=0;k<nn.count-1;k++){
        matrix_fill(nn.delta_w[k],0);
        matrix_fill(nn.delta_b[k],0);
    }
    for(int n=0;n<train_data.datasets;n++){
        nn_clean(nnG);
        matrixcpy(NN_INPUT(nn),train_data.input[n]);
        forward(nn);
        d_softmax(NN_OUTPUT(nnG),nn.weighted_sum[nn.count-1]);
        for(int i=0;i< NN_OUTPUT(nn).rows;i++){
            NN_OUTPUT(nnG).ptr[i][0] *= (train_data.output[n].ptr[i][0] - NN_OUTPUT(nn).ptr[i][0]);
        }
        Matrix transpose1 = matrix_transpose(nn.activations[nn.count-1]);
        matrix_dot_product(nnG.weights[nn.count-1], NN_OUTPUT(nnG), transpose1);
        matrixcpy(nnG.biases[nn.count-1], NN_OUTPUT(nnG));
        matrix_sum(nn.delta_w[nn.count-1], nnG.weights[nn.count-1]);
        matrix_sum(nn.delta_b[nn.count-1], nnG.biases[nn.count-1]);
        Matrix transpose2 = matrix_transpose(nn.weights[nn.count-1]);
        matrix_dot_product(nnG.activations[nn.count-1],transpose2, NN_OUTPUT(nnG));
        for(int i=0;i<nnG.activations[nn.count-1].rows;i++){
            nnG.activations[nn.count-1].ptr[i][0] *= d_ReLU(nn.weighted_sum[nn.count-2].ptr[i][0]);
        }
        Matrix transpose3 = matrix_transpose(nn.activations[nn.count-2]);
        matrix_dot_product(nnG.weights[nn.count-2],nnG.activations[nn.count-1],transpose3);
        matrixcpy(nnG.biases[nn.count-2],nnG.activations[nn.count-1]);
        matrix_sum(nn.delta_w[nn.count-2], nnG.weights[nn.count-2]);
        matrix_sum(nn.delta_b[nn.count-2], nnG.biases[nn.count-2]);

        free_matrix(transpose1);
        free_matrix(transpose2);
        free_matrix(transpose3);

    }
    for(int i=0;i<nnG.count;i++){
        for(int j=0;j<nnG.weights[i].rows;j++){
            for(int k=0;k<nnG.weights[i].cols;k++){
                nn.delta_w[i].ptr[j][k] /= (double)train_data.datasets;
            }
        }
        for(int j=0;j<nnG.biases[i].rows;j++){
            for(int k=0;k<nnG.biases[i].cols;k++){
                nn.delta_b[i].ptr[j][k] /= (double)train_data.datasets;
            }
        }
    }
    return 0;
}
void learn(Network nn, Network nnG, double learning_rate){
    for(int i=0;i<nn.count;i++){
        for(int j=0;j<nn.weights[i].rows;j++){
            for(int k=0;k<nn.weights[i].cols;k++){
                nn.weights[i].ptr[j][k] -= (nn.delta_w[i].ptr[j][k] * learning_rate);
            }
        }
        for(int j=0;j<nn.biases[i].rows;j++){
            nn.biases[i].ptr[j][0] -= (nnG.delta_b[i].ptr[j][0] * learning_rate);
        }
    }
}