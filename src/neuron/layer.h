#ifndef __LAYER__
#define __LAYER__
#include "../matrix/matrix.h"

/* Define dense neuron layer of given input type */
#define LayerDenseDefn(type)                            \
    struct {                                            \
        int ninputs, nneurons;                          \
        Matrix2DDefn(type) lweights, lbiases, loutputs; \
    }

/* Initialize dense neuron layer */
#define LayerDenseInit(self, n_inputs, n_neurons)          \
    {                                                      \
        self.ninputs = n_inputs;                           \
        self.nneurons = n_neurons;                         \
        Matrix2DInit(self.lweights, n_neurons, n_inputs);  \
        Matrix2DInit(self.lbiases, 1, n_neurons);          \
        Matrix2DInit(self.loutputs, BATCHSIZE, n_neurons); \
    }

/* Fill dense neuron layer */
#define LayerDenseFill(self, weights, biases) \
    {                                         \
        Matrix2DFill(self.lweights, weights); \
        Matrix2DFill(self.lbiases, biases);   \
    }

/* Fill dense neuron layer with random values */
#define LayerDenseRand(self)         \
    {                                \
        Matrix2DRand(self.lweights); \
        Matrix2DRand(self.lbiases);  \
    }

/* Calculate output through forward pass */
#define LayerDenseFrwd(self, l_input)                             \
    {                                                             \
        Matrix2DTrsp(self.lweights);                              \
        Matrix2DMult(l_input, self.lweights, self.loutputs);      \
        VectorRowAdd(self.loutputs, self.lbiases, self.loutputs); \
    }

/* Free dense neuron layer */
#define LayerDenseFree(self)         \
    {                                \
        Matrix2DFree(self.lweights); \
        Matrix2DFree(self.lbiases);  \
        Matrix2DFree(self.loutputs); \
    }
#endif