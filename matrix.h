//
// Created by Krzysiek on 07.06.2023.
//

#ifndef NEURALNETWORK_MATRIX_H
#define NEURALNETWORK_MATRIX_H
#include <stddef.h>
#define PRINT_MATRIX(m) matrix_print(m,#m)
#define ACTIVATE(matrix,func) apply_actvf(func,matrix,#func)
typedef struct{
    size_t rows;
    size_t cols;
    float **ptr;
}Matrix;
float randf(float min, float max);
void softmax(Matrix matrix);
void d_softmax(Matrix destination,Matrix matrix);
Matrix matrix_allocate(size_t rows, size_t cols);
void matrix_fill(Matrix matrix, float value);
void matrix_dot_product(Matrix destination, Matrix a, Matrix b);
void matrix_sum(Matrix destination, Matrix a);
void matrix_randomize(Matrix matrix, float min, float max);
void matrix_print(Matrix matrix,const char* name_of_matrix);
void matrixcpy(Matrix destination, Matrix source);
void apply_actvf(float(*f)(float), Matrix matrix, const char* name_of_func);
void matrix_multiply(Matrix matrix,float constant);
Matrix matrix_transpose(Matrix matrix);
void free_matrix(Matrix matrix);

#endif //NEURALNETWORK_MATRIX_H
