//
// Created by Krzysiek on 08.06.2023.
//
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include "mlearning.h"
#include "nnetwork.h"
#include "matrix.h"
#include "actvfunc.h"
void forward(Network nn){
    for(int i=0;i<nn.count;i++){
        matrix_dot_product(nn.activations[i+1],nn.weights[i],nn.activations[i]);
        matrix_sum(nn.activations[i+1],nn.biases[i]);
        ACTIVATE(nn.activations[i+1],Sigmoid); // keeping it as it is but would be nice to change it
                                               //Sigmoid/ReLU/LeakyReLU//
    }
    softmax(NN_OUTPUT(nn));
}
float cost(Network nn,size_t input_size,size_t output_size,size_t data_sets){
    assert(NN_INPUT(nn).rows == input_size);
    assert(NN_OUTPUT(nn).rows == output_size);
    Matrix input = matrix_allocate(input_size,1);
    Matrix output = matrix_allocate(output_size,1);
    FILE *file;
    float cost = 0.0f;
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
            float difference = NN_OUTPUT(nn).ptr[i][0] - output.ptr[i][0];
            cost += difference * difference;
        }
    }
    fclose(file);
    return cost/(float)data_sets;
}
size_t backpropagation(Network nn, Network nnG, size_t input_size,size_t output_size,size_t data_sets){
    nn_clean(nnG);
    assert(NN_INPUT(nn).rows == input_size);
    assert(NN_OUTPUT(nn).rows == output_size);
    Matrix input = matrix_allocate(input_size,1);
    Matrix output = matrix_allocate(output_size,1);
    FILE *file;
    float cost = 0.0f;
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
        //backprop for output layer
        d_softmax(NN_OUTPUT(nnG), nn.activations[nn.count]);
        for(int i=0;i<output_size;i++){
            NN_OUTPUT(nnG).ptr[i][0] *= output.ptr[i][0] - NN_OUTPUT(nn).ptr[i][0]; //can be wrong 2:31 // error calc
        }
        Matrix transposition = matrix_transpose(nn.activations[nn.count-1]);
        matrix_dot_product(nnG.weights[nn.count-1], NN_OUTPUT(nnG),transposition);
        matrixcpy(nnG.biases[nn.count-1], NN_OUTPUT(nnG));

        for(size_t layer=nn.count-1;layer>0;layer--){
            matrix_dot_product(nnG.activations[layer],nn.weights[layer], NN_OUTPUT(nnG)); //error calc
            for(size_t a=0;a<nn.activations[layer].rows;a++){
                for(size_t rows = 0;rows<nn.activations[layer].rows;rows++){
                    nnG.activations[layer].ptr[rows][0] *= d_Sigmoid(nnG.activations[layer].ptr[rows][0]); // tu jest blad
                }
            }
            transposition = matrix_transpose(nn.activations[layer-1]);
            matrix_dot_product(nnG.weights[layer-1],nnG.activations[layer],transposition);
            matrixcpy(nnG.biases[layer-1],nnG.activations[layer]);
        }
        free_matrix(transposition);
    }
    fclose(file);
    free_matrix(input);
    free_matrix(output);
    return 0;
}
void learn(Network nn, Network nnG, float learning_rate){
    for(int i=0;i<nn.count;i++){
        for(int j=0;j<nn.weights[i].rows;j++){
            for(int k=0;k<nn.weights[i].cols;k++){
                nn.weights[i].ptr[j][k] -= nnG.weights[i].ptr[j][k] * learning_rate;
            }
        }
        for(int j=0;j<nn.biases[i].rows;j++){
            nn.biases[i].ptr[j][0] -= nnG.biases[i].ptr[j][0] * learning_rate;
        }
    }
}