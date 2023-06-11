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
bool is_valid(float x)
{
    return x*0.0==0.0;
}
void forward(Network nn){
    for(int i=0;i<nn.count;i++){
        matrix_dot_product(nn.weighted_sum[i],nn.weights[i],nn.activations[i]);
        matrix_sum(nn.weighted_sum[i],nn.biases[i]);
        matrixcpy(nn.activations[i+1],nn.weighted_sum[i]);
        ACTIVATE(nn.activations[i+1],Sigmoid); // keeping it as it is but would be nice to change it
                                               //Sigmoid/ReLU/LeakyReLU//
    }
    softmax(NN_OUTPUT(nn));
}
double cost(Network nn,size_t input_size,size_t output_size,size_t data_sets){
    assert(NN_INPUT(nn).rows == input_size);
    assert(NN_OUTPUT(nn).rows == output_size);
    Matrix input = matrix_allocate(input_size,1);
    Matrix output = matrix_allocate(output_size,1);
    FILE *file;
    double cost = 0.0f;
    file = fopen("trainingdata.txt","r");
    for(int n=0;n<data_sets;n++) {
        for (int i = 0; i < input_size; i++) {
            fscanf(file,"%f", &input.ptr[i][0]);
        }
        for(int i = 0;i<output_size;i++) {
            fscanf(file, "%f", &output.ptr[i][0]);
        }
        matrixcpy(NN_INPUT(nn),input);
        forward(nn);
        for(int i=0;i<output_size;i++){
            double difference = NN_OUTPUT(nn).ptr[i][0] - output.ptr[i][0];
//            PRINT_MATRIX(NN_OUTPUT(nn));
            cost += difference * difference;
        }
    }
    fclose(file);
    return cost/(double)data_sets;
}
size_t backpropagation(Network nn, Network nnG, size_t input_size,size_t output_size,size_t data_sets){
    assert(NN_INPUT(nn).rows == input_size);
    assert(NN_OUTPUT(nn).rows == output_size);
    Matrix input = matrix_allocate(input_size,1);
    Matrix output = matrix_allocate(output_size,1);
    FILE *file;
    double cost = 0.0f;
    nn_clean(nnG);
    file = fopen("trainingdata.txt","r");
    for(int n=0;n<data_sets;n++) {
        for (int i = 0; i < input_size; i++) {
            fscanf(file,"%lf", &input.ptr[i][0]);
        }
        for(int i = 0;i<output_size;i++) {
            fscanf(file, "%lf", &output.ptr[i][0]);
        }
        for(int i=0;i<=nn.count;i++){
            matrix_fill(nnG.activations[i],0);
        }
        matrixcpy(NN_INPUT(nn),input);
        forward(nn);
        PRINT_NN(nn);
        //error output
        d_softmax(NN_OUTPUT(nnG),nn.weighted_sum[nn.count-1]);
        for(int i =0;i< NN_OUTPUT(nnG).rows;i++){
            NN_OUTPUT(nnG).ptr[i][0] *= (output.ptr[i][0] - NN_OUTPUT(nn).ptr[i][0]);
        }
        //delta output
        Matrix transpose = matrix_transpose(nn.activations[nn.count-1]);
        matrix_dot_product(nnG.weights[nn.count-1], NN_OUTPUT(nnG),transpose);
        matrixcpy(nnG.biases[nn.count-1], NN_OUTPUT(nnG));
        free_matrix(transpose);

        for(size_t i=nn.count-1;i>0;i--){ // error for hidden
            Matrix transposition1 = matrix_transpose(nn.weights[i]);
            matrix_dot_product(nnG.activations[i],transposition1,nnG.activations[i+1]);
            free_matrix(transposition1);
            for(int j=0;j<nnG.activations[i].rows;j++){
                nnG.activations[i].ptr[j][0] *= d_Sigmoid(nn.weighted_sum[i-1].ptr[j][0]);
            }
            Matrix transposition2 = matrix_transpose(nn.activations[i-1]); //delta hidden
            matrix_dot_product(nnG.weights[i-1],nnG.activations[i],transposition2);
            matrixcpy(nnG.biases[i-1],nnG.activations[i]);
            free_matrix(transposition2);
        }
    }
    fclose(file);
    free_matrix(input);
    free_matrix(output);
    for(int i=0;i<nnG.count;i++){
        for(int j=0;j<nnG.weights[i].rows;j++){
            for(int k=0;k<nnG.weights[i].cols;k++){
                nnG.weights[i].ptr[j][k] /= (double)data_sets;
            }
        }
        for(int j=0;j<nnG.biases[i].rows;j++){
            for(int k=0;k<nnG.biases[i].cols;k++){
                nnG.biases[i].ptr[j][k] /= (double)data_sets;
            }
        }
    }
    return 0;
}
void learn(Network nn, Network nnG, double learning_rate){
    for(int i=0;i<nn.count;i++){
        for(int j=0;j<nn.weights[i].rows;j++){
            for(int k=0;k<nn.weights[i].cols;k++){
                nn.weights[i].ptr[j][k] -= (nnG.weights[i].ptr[j][k] * learning_rate);
            }
        }
        for(int j=0;j<nn.biases[i].rows;j++){
            nn.biases[i].ptr[j][0] -= (nnG.biases[i].ptr[j][0] * learning_rate);
        }
    }
}