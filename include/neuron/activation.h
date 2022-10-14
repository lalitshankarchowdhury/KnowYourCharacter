#ifndef __ACTIVATION__
#define __ACTIVATION__
#include <matrix/matrix.h>

enum ActivationFunctionType {
    STEP,
    LINEAR,
    RECTIFIED_LINEAR
};

/* Step-wise function */
#define STEP_FUNC(x) (x > 0) ? 1 : 0

/* Linear function*/
#define LINEAR_FUNC(x) x

/*Rectified linear function*/
#define RECTIFIED_LINEAR_FUNC(x) (x > 0) ? x : 0

/* Calculate output by applying activation function */
#define ActivationFrwd(self, activation)                        \
    {                                                           \
        switch (activation) {                                   \
        case STEP:                                              \
            Matrix2DFunc(self.loutputs, STEP_FUNC);             \
            break;                                              \
        case LINEAR:                                            \
            Matrix2DFunc(self.loutputs, LINEAR_FUNC);           \
            break;                                              \
        case RECTIFIED_LINEAR:                                  \
            Matrix2DFunc(self.loutputs, RECTIFIED_LINEAR_FUNC); \
            break;                                              \
        }                                                       \
    }
#endif