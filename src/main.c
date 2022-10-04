#include "matrix.h"
#include <stdlib.h>

int main()
{
    Matrix2DDefn(float) x, y, xy, z;
    Matrix2DDefn(float) res;
    Matrix2DInit(x, 3, 4);
    Matrix2DInit(y, 3, 4);
    Matrix2DInit(xy, 3, 3);
    Matrix2DInit(z, 3, 3);
    Matrix2DInit(res, 3, 3);
    float inputs[3][4] = { { 1, 2, 3, 2.5 }, { 2, 5, -1, 2 }, { -1.5, 2.7, 3.3, -0.8 } };
    float weights[3][4] = { { 0.2, 0.8, -0.5, 1 }, { 0.5, -0.91, 0.26, -0.5 }, { -0.26, -0.27, 0.17, 0.87 } };
    float biases[3][3] = { { 2, 3, 0.5 }, { 2, 3, 0.5 }, { 2, 3, 0.5 } };
    Matrix2DFill(x, inputs);
    Matrix2DFill(y, weights);
    Matrix2DFill(z, biases);
    Matrix2DTrsp(y);
    Matrix2DMult(x, y, xy);
    Matrix2DAdd(xy, z, res);
    Matrix2DDisp(res, "%lf");
    Matrix2DFree(x);
    Matrix2DFree(y);
    Matrix2DFree(xy);
    Matrix2DFree(z);
    Matrix2DFree(res);
    return 0;
}
