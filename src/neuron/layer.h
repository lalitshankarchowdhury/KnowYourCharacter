#ifndef __LAYER__
#define __LAYER__
#include "../matrix/matrix.h"

#define LayerDenseDefn(type)                            \
    struct {                                            \
        int ninputs, nneurons;                          \
        Matrix2DDefn(type) lweights, lbiases, loutputs; \
    }

#define LayerDenseInit(self, n_inputs, n_neurons, l_weights, l_biases) \
    {                                                                  \
        self.ninputs = n_inputs;                                       \
        self.nneurons = n_neurons;                                     \
        Matrix2DInit(self.lweights, n_neurons, n_inputs);              \
        Matrix2DInit(self.lbiases, 1, n_neurons);                      \
        Matrix2DInit(self.loutputs, BATCHSIZE, n_neurons);             \
        Matrix2DFill(self.lweights, l_weights);                        \
        Matrix2DFill(self.lbiases, l_biases);                          \
    }

#define LayerDenseFrwd(self, l_input)                             \
    {                                                             \
        Matrix2DDefn(float) linput;                               \
        Matrix2DInit(linput, BATCHSIZE, self.ninputs);            \
        Matrix2DFill(linput, l_input);                            \
        Matrix2DTrsp(self.lweights);                              \
        Matrix2DMult(linput, self.lweights, self.loutputs);       \
        VectorRowAdd(self.loutputs, self.lbiases, self.loutputs); \
        Matrix2DFree(linput);                                     \
    }

#define LayerDenseFree(self)         \
    {                                \
        Matrix2DFree(self.lweights); \
        Matrix2DFree(self.lbiases);  \
        Matrix2DFree(self.loutputs); \
    }
#endif