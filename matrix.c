//
// Created by Krzysiek on 07.06.2023.
//
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "matrix.h"

float randf(float min, float max){
    float random = ((float) rand() / RAND_MAX);
    float range = (max-min) * random;
    return min + range;
}
void softmax(Matrix matrix){
    float sum=0;
    for(int i=0;i<matrix.rows;i++){
        for(int j=0;j<matrix.cols;j++){
            matrix.ptr[i][j] = expf(matrix.ptr[i][j]);
            sum += matrix.ptr[i][j];
        }
    }
    for(int i=0;i<matrix.rows;i++){
        for(int j=0;j<matrix.cols;j++){
            matrix.ptr[i][j] /= sum;
        }
    }
}
void d_softmax(Matrix destination,Matrix matrix){
    assert(destination.rows==matrix.rows);
    assert(destination.cols==matrix.cols);
    for(int i=0;i<matrix.rows;i++){
        for(int j=0;j<matrix.cols;j++){
            if(i==j){
                destination.ptr[i][j] += matrix.ptr[i][j]*(1-matrix.ptr[i][j]);
            }
            else{
                destination.ptr[i][j] -= matrix.ptr[i][j]*matrix.ptr[i][j];
            }
        }
    }
}
Matrix matrix_allocate(size_t rows, size_t cols){
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.ptr = (float**)calloc(matrix.rows,sizeof(*matrix.ptr));
    for(int i=0;i<matrix.rows;i++){
        matrix.ptr[i] = (float*)calloc(matrix.cols , sizeof(**matrix.ptr));
    }
    for(int i=0;i<matrix.rows;i++){
        assert(matrix.ptr[i] != NULL);
    }
    assert(matrix.ptr!=NULL);
    return matrix;
}
void matrix_fill(Matrix matrix, float value){
    size_t i=0,j=0;
    for(i=0;i<matrix.rows;i++){
        for(j=0;j<matrix.cols;j++){
            matrix.ptr[i][j] = value;
        }
    }
}
void matrix_dot_product(Matrix destination, Matrix a, Matrix b){
    assert(a.cols == b.rows);
    assert(destination.rows  == a.rows);
    assert(destination.cols == b.cols);
    size_t i=0,j=0,k=0;
    size_t n = a.cols;
    for(i = 0;i < destination.rows; i++){
        for(j = 0; j < destination.cols; j++){
            destination.ptr[i][j] = 0;
            for(k = 0 ;k < n ;k++){
                destination.ptr[i][j] += a.ptr[i][k] * b.ptr[k][j];
            }
        }
    }
}
void matrix_sum(Matrix destination, Matrix a) {
    assert(destination.rows == a.rows);
    assert(destination.cols == a.cols);
    size_t i = 0, j = 0;
    for (i = 0; i < destination.rows; i++) {
        for (j = 0; j < destination.cols; j++) {
            destination.ptr[i][j] += a.ptr[i][j];
        }
    }
}
void matrix_randomize(Matrix matrix, float min, float max){
    size_t i=0,j=0;
    for(i=0;i<matrix.rows;i++){
        for(j=0;j<matrix.cols;j++){
            matrix.ptr[i][j] = randf(min,max);
        }
    }
}
void matrix_print(Matrix matrix,const char* name_of_matrix) {
    int maxDigits = 0;
    // Znajdowanie najdłuższego elementu w tablicy
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            int digits = snprintf(NULL, 0, "%f", matrix.ptr[i][j]);
            if (digits > maxDigits) {
                maxDigits = digits;
            }
        }

    }
    printf("%s\n", name_of_matrix);
    for (int i = 0; i < matrix.rows; i++) {
        printf("[ ");
        for (int j = 0; j < matrix.cols; j++) {
            printf("%-*.*f ", maxDigits , 6, matrix.ptr[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
}
void matrixcpy(Matrix destination, Matrix source){
    assert(source.rows == destination.rows);
    assert(source.cols == destination.cols);
    size_t i=0,j=0;
    for(i=0;i<destination.rows;i++){
        for(j=0;j<destination.cols;j++){
            destination.ptr[i][j] = source.ptr[i][j];
        }
    }
}
void apply_actvf(float(*actvfunc)(float),Matrix matrix,const char* name_of_func){
//    printf("Applying: %s\n",name_of_func); // func name, testing only
    for(int i=0;i<matrix.rows;i++){
        for(int j=0;j<matrix.cols;j++){
            matrix.ptr[i][j] = actvfunc(matrix.ptr[i][j]);
        }
    }
}
void free_matrix(Matrix matrix){
    for(int i=0;i<matrix.rows;i++){
        free(matrix.ptr[i]);
    }
    free(matrix.ptr);
}
void matrix_multiply(Matrix matrix,float constant){
    for(int i=0;i<matrix.rows;i++){
        for(int j=0;j<matrix.cols;j++){
            matrix.ptr[i][j]*=constant;
        }
    }
}
Matrix matrix_transpose(Matrix matrix){
    assert(matrix.ptr != NULL);
    Matrix transpositon = matrix_allocate(matrix.cols,matrix.rows);
    for(int i=0;i<transpositon.rows;i++){
        for(int j=0;j<transpositon.cols;j++){
            transpositon.ptr[i][j] = matrix.ptr[j][i];
        }
    }
    return transpositon;
}
