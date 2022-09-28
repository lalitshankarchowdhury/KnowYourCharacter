#ifndef __MATRIX__
#define __MATRIX__
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* Define 2D matrix of a given type */
#define Matrix2DDefn(type) \
    struct _Matrix2D {     \
        int nrows, ncols;  \
        type** data;       \
    }

/* Initialize matrix with given row and column size */
#define Matrix2DInit(self, rowsize, colsize) ({                                  \
    self.nrows = rowsize;                                                        \
    self.ncols = colsize;                                                        \
    assert((self.data = calloc(rowsize, sizeof(*self.data))) != NULL);           \
    for (int i = 0; i < rowsize; i++) {                                          \
        assert((self.data[i] = calloc(colsize, sizeof(*self.data[0]))) != NULL); \
    }                                                                            \
})

/* Fill matrix with given elements */
#define Matrix2DFill(self, array) ({           \
    for (int i = 0; i < self.nrows; i++) {     \
        for (int j = 0; j < self.ncols; j++) { \
            self.data[i][j] = array[i][j];     \
        }                                      \
    }                                          \
})

/* Display 2D matrix */
#define Matrix2DDisp(self, format) ({             \
    for (int i = 0; i < self.nrows; i++) {        \
        for (int j = 0; j < self.ncols; j++) {    \
            printf(format "\t", self.data[i][j]); \
        }                                         \
        putchar('\n');                            \
    }                                             \
})

/* Add two 2D matrices */
#define Matrix2DAdd(mat1, mat2, result) ({                         \
    assert(mat1.nrows == result.nrows);                            \
    assert(mat1.ncols == result.ncols);                            \
    assert(mat2.nrows == result.nrows);                            \
    assert(mat2.ncols == result.ncols);                            \
    for (int i = 0; i < result.nrows; i++) {                       \
        for (int j = 0; j < result.ncols; j++) {                   \
            result.data[i][j] = mat1.data[i][j] + mat2.data[i][j]; \
        }                                                          \
    }                                                              \
})

/* Multiply two 2D matrices */
#define Matrix2DMult(mat1, mat2, result) ({               \
    assert(mat1.ncols == mat2.nrows);                     \
    assert(result.nrows == mat1.nrows);                   \
    assert(result.ncols == mat2.ncols);                   \
    assert(mat1.data != NULL);                            \
    assert(mat2.data != NULL);                            \
    assert(result.data != NULL);                          \
    for (int i = 0; i < result.nrows; i++) {              \
        for (int j = 0; j < result.ncols; j++) {          \
            typeof(**result.data) sum = { 0 };            \
            for (int k = 0; k < mat1.ncols; k++) {        \
                assert(mat1.data[i] != NULL);             \
                assert(mat2.data[k] != NULL);             \
                sum += mat1.data[i][k] * mat2.data[k][j]; \
            }                                             \
            assert(result.data[i] != NULL);               \
            result.data[i][j] = sum;                      \
        }                                                 \
    }                                                     \
})

/* Transpose 2D matrix */
#define Matrix2DTrsp(self) ({                                                      \
    assert(self.data != NULL);                                                     \
    typeof(self.data) new_data;                                                    \
    assert((new_data = malloc(self.ncols * sizeof(*new_data))) != NULL);           \
    for (int j = 0; j < self.ncols; j++) {                                         \
        assert((new_data[j] = malloc(self.nrows * sizeof(*new_data[0]))) != NULL); \
    }                                                                              \
    for (int i = 0; i < self.nrows; i++) {                                         \
        for (int j = 0; j < self.ncols; j++) {                                     \
            assert(self.data[i] != NULL);                                          \
            new_data[j][i] = self.data[i][j];                                      \
        }                                                                          \
    }                                                                              \
    int old_nrows = self.nrows;                                                    \
    int old_ncols = self.ncols;                                                    \
    Matrix2DFree(self);                                                            \
    self.nrows = old_ncols;                                                        \
    self.ncols = old_nrows;                                                        \
    self.data = new_data;                                                          \
})

/* Free memory allocated to matrix */
#define Matrix2DFree(self) ({              \
    assert(self.data != NULL);             \
    for (int i = 0; i < self.nrows; i++) { \
        assert(self.data[i] != NULL);      \
        free(self.data[i]);                \
    }                                      \
    free(self.data);                       \
    *(&self.data) = NULL;                  \
})
#endif
