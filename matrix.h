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
    double **ptr;
}Matrix;
double randf(double min, double max);
void softmax(Matrix matrix);
void d_softmax(Matrix destination,Matrix matrix);
Matrix matrix_allocate(size_t rows, size_t cols);
void matrix_fill(Matrix matrix, double value);
void matrix_dot_product(Matrix destination, Matrix a, Matrix b);
void matrix_sum(Matrix destination, Matrix a);
void matrix_randomize(Matrix matrix, double min, double max);
void matrix_print(Matrix matrix,const char* name_of_matrix);
void matrixcpy(Matrix destination, Matrix source);
void apply_actvf(double(*f)(double), Matrix matrix, const char* name_of_func);
void matrix_multiply(Matrix matrix,double constant);
Matrix matrix_transpose(Matrix matrix);
size_t max_value_index_vector(Matrix matrix);

void free_matrix(Matrix matrix);

#endif //NEURALNETWORK_MATRIX_H
