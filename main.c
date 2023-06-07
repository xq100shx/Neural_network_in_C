#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

int main() {
    srand(time(NULL));
////TESTS
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
////

    return 0;
}
