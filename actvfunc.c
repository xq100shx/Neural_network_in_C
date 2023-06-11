//
// Created by Krzysiek on 07.06.2023.
//
#include <math.h>
#include "actvfunc.h"

double Sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
double d_Sigmoid(double x){
    return Sigmoid(x)*(1- Sigmoid(x));
}
double ReLU(double x){    //im not sure if it's correct
    if(x>0) return x;
    else return 0.0f;
}
double d_ReLU(double x){  //im not sure if it's correct
    if(x>0) return 1.0f;
    else return 0.0f;
}
double Leaky_ReLU(double x){  //im not sure if it's correct
    if(x>0) return x;
    else return 0.01f*x;
}
double d_Leaky_ReLU(double x){    //im not sure if it's correct
    if(x>0) return 1.0f;
    else return 0.01f;
}
//note add matrix interpretations of functions to matrix lib
