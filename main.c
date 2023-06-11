#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>
#include "matrix.h"
#include "actvfunc.h"
#include "nnetwork.h"
#include "mlearning.h"

int unit_testing();

int main() {
//    size_t architecture[] = {2, 2, 2};
//    size_t layers = sizeof(architecture) / sizeof(architecture[0]);
//    size_t input_size = architecture[0];
//    size_t output_size = architecture[layers - 1];
//    size_t data_sets = 100;
//    double learning_rate = 0.25f;

    srand(time(NULL));
//    Network nn = nn_allocate(layers, architecture);
//    Network nnG = nn_allocate(layers, architecture);
//    printf("cost = %f\n", cost(nn, input_size, output_size, data_sets));
//    nn_randomize(nn, -1, 1);
//    for(int i=0;i<100;i++) {
//        printf("cost = %f\n", cost(nn, input_size, output_size, data_sets));
//        backpropagation(nn, nnG, input_size, output_size, data_sets);
//        learn(nn,nnG,learning_rate);
//    }
//    free_network(nn);
//    free_network(nnG);
    unit_testing();
    return 0;
}
int unit_testing() {
    float x = 10;
    size_t architecture[] = {2, 4, 2};
    size_t layers = sizeof(architecture) / sizeof(architecture[0]);
    size_t input_size = 2;
    size_t output_size = 2;
    size_t data_sets = 100;
    double learning_rate = 0.25f;
#if 0 //randf() test
    printf("%d\n",RAND_MAX);
    for(size_t i=0;i<5;i++) {
        printf("[%zu] random float value: %f\n", i,randf(0, 3));
    }
#endif
#if 0 //matrix_allocate test
    Matrix test = matrix_allocate(3,3);
    PRINT_MATRIX(test);
    free_matrix(test);
#endif
#if 0 // matrix fill test
    Matrix test = matrix_allocate(3,3);
    matrix_fill(test,3);
    PRINT_MATRIX(test);
    free_matrix(test);
#endif
#if 0 //matrix_dot_product test
    Matrix testA = matrix_allocate(3,3);
    Matrix testB = matrix_allocate(3,3);
    Matrix testDst = matrix_allocate(3,3);
    matrix_fill(testA,3);
    matrix_fill(testB,4);
    matrix_dot_product(testDst,testA,testB);
    PRINT_MATRIX(testA);
    PRINT_MATRIX(testB);
    PRINT_MATRIX(testDst);
    free_matrix(testA);
    free_matrix(testB);
    free_matrix(testDst);
#endif
#if 0 //matrix_sum test
    Matrix testA = matrix_allocate(3,3);
    Matrix testDst = matrix_allocate(3,3);
    matrix_fill(testA,3);
    matrix_fill(testDst,12);
    printf("Before adding:\n");
    PRINT_MATRIX(testA);
    PRINT_MATRIX(testDst);
    printf("After adding:\n");
    matrix_sum(testDst,testA);
    PRINT_MATRIX(testDst);
    free_matrix(testA);
    free_matrix(testDst);
#endif
#if 0 //matrix_randomize test
    Matrix test = matrix_allocate(5,4);
    float min = 0.0f;
    float max = 1000.0f;
    printf("Range: [%f , %f]\n",min,max);
    matrix_randomize(test,min,max);
    PRINT_MATRIX(test);
    free_matrix(test);
#endif
#if 0 //matrixcpy test
    Matrix test = matrix_allocate(3,3);
    Matrix testDst = matrix_allocate(3,3);
    matrix_fill(test,3);
    matrix_fill(testDst,11);
    printf("Before copying:\n");
    PRINT_MATRIX(test);
    PRINT_MATRIX(testDst);
    printf("After copying:\n");
    matrixcpy(testDst,test);
    PRINT_MATRIX(testDst);
    free_matrix(test);
    free_matrix(testDst);
#endif
#if 0 //actv_apply test
    Matrix test = matrix_allocate(5,1);
    matrix_randomize(test,-5,5);
    printf("Input:\n");
    PRINT_MATRIX(test);
    ACTIVATE(test,Sigmoid);
    printf("Result:\n");
    PRINT_MATRIX(test);
    matrix_randomize(test,-5,5);
    printf("Input:\n");
    PRINT_MATRIX(test);
    ACTIVATE(test,d_Sigmoid);
    printf("Result:\n");
    PRINT_MATRIX(test);
    matrix_randomize(test,-5,5);
    printf("Input:\n");
    PRINT_MATRIX(test);
    ACTIVATE(test,ReLU);
    printf("Result:\n");
    PRINT_MATRIX(test);
    matrix_randomize(test,-5,5);
    printf("Input:\n");
    PRINT_MATRIX(test);
    ACTIVATE(test,d_ReLU);
    printf("Result:\n");
    PRINT_MATRIX(test);
    matrix_randomize(test,-5,5);
    printf("Input:\n");
    PRINT_MATRIX(test);
    ACTIVATE(test,Leaky_ReLU);
    printf("Result:\n");
    PRINT_MATRIX(test);
    matrix_randomize(test,-5,5);
    printf("Input:\n");
    PRINT_MATRIX(test);
    ACTIVATE(test,d_Leaky_ReLU);
    printf("Result:\n");
    PRINT_MATRIX(test);
    free_matrix(test);
#endif
#if 0 //softmax test
    Matrix test = matrix_allocate(5,1);
    matrix_randomize(test,0,1);
    printf("Before softmax:\n");
    PRINT_MATRIX(test);
    printf("After softmax:\n");
    softmax(test);
    PRINT_MATRIX(test);
    free_matrix(test);
#endif
#if 0 //d_softmax test
    Matrix test = matrix_allocate(5,1);
    Matrix derivative = matrix_allocate(5,1);
    matrix_randomize(test,0,1);
    printf("Input:\n");
    PRINT_MATRIX(test);
    printf("Derivative:\n");
    d_softmax(derivative,test);
    PRINT_MATRIX(derivative);
    free_matrix(test);
    free_matrix(derivative);
#endif
#if 0 //sigmoid test
    printf("sigmoid = %f\n",sigmoid(x));
#endif
#if 0 //d_sigmoid test
    printf("derivative of sigmoid = %f\n",d_sigmoid(x));
#endif
#if 0 // matrix_multiply test
    Matrix test = matrix_allocate(3,3);
    matrix_fill(test,3);
    PRINT_MATRIX(test);
    printf("Multiplying...\n");
    matrix_multiply(test,3);
    PRINT_MATRIX(test);
    free_matrix(test);
#endif
#if 0 // matrix_transposition test
    Matrix test = matrix_allocate(5,4);
    float min = 0.0f;
    float max = 100.0f;
    matrix_randomize(test,min,max);
    PRINT_MATRIX(test);
    printf("Transposing...\n");
    Matrix transposed = matrix_transpose(test);
    PRINT_MATRIX(transposed);
    free_matrix(transposed);
    free_matrix(test);
#endif
#if 0 //ReLU test
    printf("ReLU = %f\n",ReLU(x));
#endif
#if 0 //d_ReLU test
    printf("derivative of ReLU = %f\n",d_ReLU(x));
#endif
#if 0 //LeakyReLU test
    printf("Leaky ReLU = %f\n",Leaky_ReLu(x));
#endif
#if 0 //d_LeakyReLU test
    printf("derivtive of Leaky ReLU = %f\n",d_Leaky_ReLu(x));
#endif
#if 0 //nn_allocate + nn_print test
    Network test = nn_allocate(layers,architecture);
    PRINT_NN(test);
    free_network(test);
#endif
#if 0 //nn_randomize test
    Network test = nn_allocate(layers,architecture);
    nn_randomize(test,-5,5);
    PRINT_NN(test);
    free_network(test);
#endif
#if 0 //nn_clean test
    Network test = nn_allocate(layers,architecture);
    nn_randomize(test,-5,5);
    PRINT_NN(test);
    printf("cleaning...\n");
    nn_clean(test);
    PRINT_NN(test);
    free_network(test);
#endif
#if 0 //save_values test
    Network test = nn_allocate(layers,architecture);
    nn_randomize(test,-5,5);
    PRINT_NN(test);
    printf("saving...\n");
    save_values(test);
    free_network(test);
#endif
#if 0 // forward test
    Network test = nn_allocate(layers,architecture);
    nn_randomize(test,-5,5);
    printf("Input neural network:\n");
    PRINT_NN(test);
    printf("Forwarding...\n");
    forward(test);
    PRINT_NN(test);
    free_network(test);
#endif
#if 0 //cost test
    Network test = nn_allocate(layers, architecture);
    for(int i=0;i<10;i++){
        nn_randomize(test, -1, 1);
        printf("avg cost = %f\n", cost(test, input_size, output_size, data_sets));
}
    free_network(test);
#endif
#if 1 //backpropagation & learning test
    Network test = nn_allocate(layers, architecture);
    Network testG = nn_allocate(layers, architecture);
    nn_randomize(test, -1, 1);

    printf("avg cost = %f\n", cost(test, input_size, output_size, data_sets));
    for(int i=0;i<1000;i++){
        backpropagation(test,testG,input_size, output_size, data_sets);
        learn(test,testG,learning_rate);
}
    printf("avg cost = %f\n", cost(test, input_size, output_size, data_sets));
    free_network(test);
    free_network(testG);
#endif
#if 0 // file
    FILE *file;

    file = fopen("trainingdata.txt","r");
    int in=2;
    int data_sets = 2;
    int out = 1;
    float x[in][1];
    float buffer[out][1];
    for(int n=0;n<data_sets;n++) {
        for (int i = 0; i < in; i++) {
            fscanf(file, "%f", &x[i][0]);
        }
        for(int i = 0;i<out;i++) {
            fscanf(file, "%f", &buffer[i][0]);
        }

        for (int i = 0; i < in; i++) {
            printf("%f\n",x[i][0]);
        }
        for(int i = 0;i<out;i++) {
            printf("%f\n",buffer[i][0]);
        }
        printf("\n");

    }
    fclose(file);
#endif
    return 0;
}