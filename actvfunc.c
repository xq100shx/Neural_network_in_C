//
// Created by Krzysiek on 07.06.2023.
//
#include <math.h>
#include "actvfunc.h"

float Sigmoid(float x){
    return 1.0f/(1.0f+expf(-x));
}
float d_Sigmoid(float x){
    return Sigmoid(x)*(1- Sigmoid(x));
}
float ReLU(float x){    //im not sure if it's correct
    if(x>0) return x;
    else return 0.0f;
}
float d_ReLU(float x){  //im not sure if it's correct
    if(x>0) return 1.0f;
    else return 0.0f;
}
float Leaky_ReLU(float x){  //im not sure if it's correct
    if(x>0) return x;
    else return 0.01f*x;
}
float d_Leaky_ReLU(float x){    //im not sure if it's correct
    if(x>0) return 1.0f;
    else return 0.01f;
}
//note add matrix interpretations of functions to matrix lib
