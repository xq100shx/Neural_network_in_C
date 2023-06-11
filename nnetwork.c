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
    nn.weighted_sum = calloc(nn.count,sizeof(*nn.weighted_sum));
    assert(nn.weighted_sum != NULL);
    nn.weights = calloc(nn.count,sizeof(*nn.weights));
    assert(nn.weights != NULL);
    nn.biases = calloc(nn.count,sizeof(*nn.biases));
    assert(nn.biases!= NULL);
    nn.delta_w = calloc(nn.count,sizeof(*nn.weights));
    assert(nn.delta_w != NULL);
    nn.delta_b = calloc(nn.count,sizeof(*nn.biases));
    assert(nn.delta_b!= NULL);
    for(int i=0;i<nn.count;i++){
        nn.activations[i] = matrix_allocate(architecture[i],1);
        nn.weighted_sum[i] = matrix_allocate(architecture[i+1],1);
        nn.weights[i] = matrix_allocate(architecture[i+1],nn.activations[i].rows);
        nn.biases[i] = matrix_allocate(architecture[i+1],1);
        nn.delta_w[i] = matrix_allocate(architecture[i+1],nn.activations[i].rows);
        nn.delta_b[i] = matrix_allocate(architecture[i+1],1);
    }
    nn.activations[nn.count] = matrix_allocate(architecture[nn.count],1);
    return nn;
}
void nn_randomize(Network nn ,double min,double max){
    size_t i=0,j=0;
    for(i=0;i<nn.count;i++){
        matrix_randomize(nn.weights[i],min,max);
        matrix_randomize(nn.biases[i],min,max);
    }
}
void nn_print(Network nn , char* name){
    size_t i=0,j=0;
    char buf[256];
    printf("%s\n[\n",name);
    snprintf(buf,sizeof(buf),"Input weights");
    matrix_print(nn.weights[0], buf);
    snprintf(buf,sizeof(buf),"Input");
    matrix_print(NN_INPUT(nn),buf);
    snprintf(buf,sizeof(buf),"Input biases");
    matrix_print(nn.biases[0], buf);
    snprintf(buf,sizeof(buf),"Input Weighted sum");
    matrix_print(nn.weighted_sum[0],buf);
    if(nn.count>1) {
        for (i = 1; i < nn.count; i++) {
            snprintf(buf, sizeof(buf), "Weights %zu", i);
            matrix_print(nn.weights[i], buf);
            snprintf(buf, sizeof(buf), "Activation %zu", i);
            matrix_print(nn.activations[i], buf);
            snprintf(buf, sizeof(buf), "Biases %zu", i);
            matrix_print(nn.biases[i], buf);
            snprintf(buf, sizeof(buf), "Weighted sum %zu", i);
            matrix_print(nn.weighted_sum[i], buf);
        }
    }
    snprintf(buf,sizeof(buf),"Output");
    matrix_print(NN_OUTPUT(nn),buf);
    printf("]\n");
}
void nn_clean(Network nn){
    for(size_t i=0;i<nn.count;i++){
        matrix_fill(nn.weights[i],0);
        matrix_fill(nn.biases[i],0);
        matrix_fill(nn.delta_w[i],0);
        matrix_fill(nn.delta_b[i],0);
        matrix_fill(nn.activations[i],0);
        matrix_fill(nn.weighted_sum[i],0);
    }
    matrix_fill(NN_OUTPUT(nn),0);
}
void load_network(Network nn){
    FILE *file;
    file = fopen("network.txt","r");
    for(int n=0;n<nn.count;n++) {
        for(int i=0;i<nn.weights[n].rows;i++){
            for(int j=0;j<nn.weights[n].cols;j++){
                fscanf(file,"%lf",&nn.weights[n].ptr[i][j]);
            }
        }
        for(int i=0;i<nn.biases[n].rows;i++){
            for(int j=0;j<nn.biases[n].cols;j++){
                fscanf(file,"%lf",&nn.biases[n].ptr[i][j]);
            }
        }
    }
    fclose(file);
}
TData td_allocate(size_t in_count,size_t out_count,size_t datasets){
    TData training_data;
    training_data.datasets = datasets;
    training_data.in_count = in_count;
    training_data.out_count = out_count;
    training_data.input = calloc(datasets,sizeof(*training_data.input));
    training_data.output = calloc(datasets,sizeof(*training_data.output));
    for(int i=0;i<datasets;i++){
        training_data.input[i] = matrix_allocate(in_count,1);
        training_data.output[i] = matrix_allocate(out_count,1);
    }
    return training_data;
}
void pass_data(TData train){
    FILE *file;
    file = fopen("trainingdata.txt","r");
    for(int n=0;n<train.datasets;n++) {
        for (int i = 0; i < train.in_count; i++) {
            fscanf(file,"%lf", &train.input[n].ptr[i][0]);
        }
        for (int i = 0; i < train.out_count; i++) {
            fscanf(file,"%lf", &train.output[n].ptr[i][0]);
        }
    }
    fclose(file);
}
void td_print(TData t_data , char* name){
    size_t i=0,j=0;
    char buf[256];
    printf("%s\n[\n",name);
        for (i = 0; i < t_data.datasets; i++) {
            snprintf(buf, sizeof(buf), "Data input %zu", i);
            matrix_print(t_data.input[i], buf);
            snprintf(buf, sizeof(buf), "Data output %zu", i);
            matrix_print(t_data.output[i], buf);
        }
    printf("]\n");
}
void free_td(TData training_d){
    size_t i=0;
    for(i=0;i<training_d.datasets;i++){
        free_matrix(training_d.input[i]);
        free_matrix(training_d.output[i]);
    }
    free(training_d.input);
    free(training_d.output);
}
void free_network(Network nn){
    size_t i=0,j=0;
    for(i=0;i<nn.count;i++){
        free_matrix(nn.activations[i]);
        free_matrix(nn.weighted_sum[i]);
        free_matrix(nn.weights[i]);
        free_matrix(nn.biases[i]);
        free_matrix(nn.delta_w[i]);
        free_matrix(nn.delta_b[i]);
    }
    free_matrix(NN_OUTPUT(nn));
    free(nn.activations);
    free(nn.weighted_sum);
    free(nn.weights);
    free(nn.biases);
    free(nn.delta_w);
    free(nn.delta_b);
}
void save_values(Network nn){
    FILE *file;
    size_t i=0,j=0,k=0;
    file = fopen("values.txt","w");
    fprintf(file, "Input weights  = [\n");
    for (j = 0; j < nn.weights[0].rows; j++) {
        for (k = 0; k < nn.weights[0].cols; k++) {
            fprintf(file, "%lf ", nn.weights[i].ptr[j][k]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n]\n");
    fprintf(file,"Input Biases = [\n");
    for (j = 0; j < nn.biases[0].rows; j++) {
        for (k = 0; k < nn.biases[0].cols; k++) {
            fprintf(file, "%lf ", nn.biases[i].ptr[j][k]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n]\n");
    for(i =1;i<nn.count;i++) {
        fprintf(file, "Weights %zu = [\n", i);
        for (j = 0; j < nn.weights[i].rows; j++) {
            for (k = 0; k < nn.weights[i].cols; k++) {
                fprintf(file, "%lf ", nn.weights[i].ptr[j][k]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n]\n");
        fprintf(file,"Biases %zu = [\n",i);
        for (j = 0; j < nn.biases[i].rows; j++) {
            for (k = 0; k < nn.biases[i].cols; k++) {
                fprintf(file, "%lf ", nn.biases[i].ptr[j][k]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n]\n");
    }
    fclose(file);
}