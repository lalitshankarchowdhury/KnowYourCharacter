#include "matrix/matrix.h"
#include "neuron/layer.h"
#include <stdlib.h>
#define BATCHSIZE 5

int main()
{
    float X[BATCHSIZE][4] = { { 1, 2, 3, 2.5 }, { 2, 5, -1, 2 }, { -1.5, 2.7, 3.3, -0.8 }, { 4.32, 3.31, 2.46, -1.53 }, { 1.7, 4.4, 3.2, 2.9 } };
    float weights1[5][4] = { { 0.2, 0.8, -0.5, 1 }, { 0.5, -0.91, 0.26, -0.5 }, { -0.2, -0.37, 0.86, 0.95 }, { 0.26, 0.44, -0.32, 0.21 }, { 0.65, -0.67, 0.43, 0.42 } };
    float weights2[2][5] = { { 0.87, 0.64, -0.32, 0.61, 0.77 }, { -0.19, -0.54, 0.1, 0.4 } };
    float biases1[1][5] = { { 2, 3, 0.5, 0.7, -0.83 } };
    float biases2[1][2] = { { 0.8, 0.6 } };
    Matrix2DDefn(float) inputs1;
    Matrix2DInit(inputs1, BATCHSIZE, 4);
    Matrix2DFill(inputs1, X);
    LayerDenseDefn(float) layer1, layer2;
    LayerDenseInit(layer1, 4, 5, weights1, biases1);
    LayerDenseInit(layer2, 5, 2, weights2, biases2);
    LayerDenseFrwd(layer1, inputs1);
    puts("Layer 1 outputs:");
    Matrix2DDisp(layer1.loutputs, "%f");
    LayerDenseFrwd(layer2, layer1.loutputs);
    puts("Layer 2 outputs:");
    Matrix2DDisp(layer2.loutputs, "%f");
    Matrix2DFree(inputs1);
    LayerDenseFree(layer1);
    return EXIT_SUCCESS;
}
