//
// Created by Krzysiek on 07.06.2023.
//
#include <math.h>
#include "actvfunc.h"

float sigmoid(float x){
    return 1.0f/(1.0f+expf(-x));
}
float d_sigmoid(float x){
    return sigmoid(x)*(1- sigmoid(x));
}
float ReLU(float x){    //im not sure if it's correct
    if(x>0) return x;
    else return 0.0f;
}
float d_ReLU(float x){  //im not sure if it's correct
    if(x>0) return 1.0f;
    else return 0.0f;
}
float Leaky_ReLu(float x){  //im not sure if it's correct
    if(x>0) return x;
    else return 0.01f*x;
}
float d_Leaky_ReLu(float x){    //im not sure if it's correct
    if(x>0) return 1.0f;
    else return 0.01f;
}
//note add matrix interpretations of functions to matrix lib
