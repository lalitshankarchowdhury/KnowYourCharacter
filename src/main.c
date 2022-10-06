#include "matrix/matrix.h"
#include "neuron/layer.h"
#include <stdlib.h>
#define BATCHSIZE 3

int main()
{
    float X[BATCHSIZE][4] = { { 1, 2, 3, 2.5 }, { 2, 5, -1, 2 }, { -1.5, 2.7, 3.3, -0.8 } };
    float weights1[3][4] = { { 0.2, 0.8, -0.5, 1 }, { 0.5, -0.91, 0.26, -0.5 }, { -0.26, -0.27, 0.17, 0.87 } };
    float biases1[1][3] = { { 2, 3, 0.5 } };
    LayerDenseDefn(float) layer1;
    LayerDenseInit(layer1, 4, 3, weights1, biases1);
    puts("Layer 1 weights:");
    Matrix2DDisp(layer1.lweights, "%f");
    puts("Layer 1 biases:");
    Matrix2DDisp(layer1.lbiases, "%f");
    LayerDenseFrwd(layer1, X);
    puts("Layer 1 outputs:");
    Matrix2DDisp(layer1.loutputs, "%f");
    LayerDenseFree(layer1);
    return EXIT_SUCCESS;
}
