#include <matrix/matrix.h>
#include <neuron/layer.h>
#include <stdlib.h>
#define BATCHSIZE 5

int main()
{
    Matrix2DDefn(float) inputs1;
    Matrix2DInit(inputs1, BATCHSIZE, 4);
    Matrix2DRand(inputs1);
    LayerDenseDefn(float) layer1, layer2;
    LayerDenseInit(layer1, 4, 5);
    LayerDenseInit(layer2, 5, 2);
    LayerDenseRand(layer1);
    LayerDenseRand(layer2);
    LayerDenseFrwd(layer1, inputs1);
    puts("Layer 1 outputs:");
    Matrix2DDisp(layer1.loutputs, "%f");
    LayerDenseFrwd(layer2, layer1.loutputs);
    puts("Layer 2 outputs:");
    Matrix2DDisp(layer2.loutputs, "%f");
    Matrix2DFree(inputs1);
    LayerDenseFree(layer1);
    LayerDenseFree(layer2);
    return EXIT_SUCCESS;
}
