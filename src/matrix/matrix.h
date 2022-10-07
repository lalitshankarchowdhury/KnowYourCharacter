#ifndef __MATRIX__
#define __MATRIX__
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* Define 2D matrix of a given type */
#define Matrix2DDefn(type) \
    struct {               \
        int nrows, ncols;  \
        type** data;       \
    }

/* Initialize 2D matrix */
#define Matrix2DInit(self, rowsize, colsize)                                         \
    {                                                                                \
        self.nrows = rowsize;                                                        \
        self.ncols = colsize;                                                        \
        assert((self.data = calloc(rowsize, sizeof(*self.data))) != NULL);           \
        for (int i = 0; i < rowsize; i++) {                                          \
            assert((self.data[i] = calloc(colsize, sizeof(*self.data[0]))) != NULL); \
        }                                                                            \
    }

/* Fill 2D matrix */
#define Matrix2DFill(self, array)                  \
    {                                              \
        assert(self.data != NULL);                 \
        for (int i = 0; i < self.nrows; i++) {     \
            for (int j = 0; j < self.ncols; j++) { \
                assert(self.data[i] != NULL);      \
                self.data[i][j] = array[i][j];     \
            }                                      \
        }                                          \
    }

/* Full 2D matrix with random values */
#define Matrix2DRand(self)                                  \
    {                                                       \
        assert(self.data != NULL);                          \
        for (int i = 0; i < self.nrows; i++) {              \
            for (int j = 0; j < self.ncols; j++) {          \
                assert(self.data[i] != NULL);               \
                self.data[i][j] = rand() / (float)RAND_MAX; \
            }                                               \
        }                                                   \
    }

/* Display 2D matrix */
#define Matrix2DDisp(self, format)                    \
    {                                                 \
        assert(self.data != NULL);                    \
        for (int i = 0; i < self.nrows; i++) {        \
            for (int j = 0; j < self.ncols; j++) {    \
                assert(self.data[i] != NULL);         \
                printf(format "\t", self.data[i][j]); \
            }                                         \
            putchar('\n');                            \
        }                                             \
    }

/* Add a row vector to 2D matrix */
#define VectorRowAdd(mat, vec, result)                               \
    {                                                                \
        assert(mat.nrows == result.nrows);                           \
        assert(mat.ncols == result.ncols);                           \
        assert(vec.ncols == mat.ncols);                              \
        assert(mat.data != NULL);                                    \
        assert(vec.data != NULL);                                    \
        assert(result.data != NULL);                                 \
        for (int i = 0; i < result.nrows; i++) {                     \
            for (int j = 0; j < result.ncols; j++) {                 \
                assert(mat.data[i] != NULL);                         \
                assert(vec.data[0] != NULL);                         \
                assert(result.data[i] != NULL);                      \
                result.data[i][j] = mat.data[i][j] + vec.data[0][j]; \
            }                                                        \
        }                                                            \
    }

/* Add a column vector to 2D matrix */
#define VectorColAdd(mat, vec, result)                               \
    {                                                                \
        assert(mat.nrows == result.nrows);                           \
        assert(mat.ncols == result.ncols);                           \
        assert(vec.nrows == mat.nrows);                              \
        assert(mat.data != NULL);                                    \
        assert(vec.data != NULL);                                    \
        assert(result.data != NULL);                                 \
        for (int i = 0; i < result.nrows; i++) {                     \
            for (int j = 0; j < result.ncols; j++) {                 \
                assert(mat.data[i] != NULL);                         \
                assert(vec.data[i] != NULL);                         \
                assert(result.data[i] != NULL);                      \
                result.data[i][j] = mat.data[i][j] + vec.data[i][0]; \
            }                                                        \
        }                                                            \
    }

/* Add two 2D matrices */
#define Matrix2DAdd(mat1, mat2, result)                                \
    {                                                                  \
        assert(mat1.nrows == result.nrows);                            \
        assert(mat1.ncols == result.ncols);                            \
        assert(mat2.nrows == result.nrows);                            \
        assert(mat2.ncols == result.ncols);                            \
        assert(mat1.data != NULL);                                     \
        assert(mat2.data != NULL);                                     \
        assert(result.data != NULL);                                   \
        for (int i = 0; i < result.nrows; i++) {                       \
            for (int j = 0; j < result.ncols; j++) {                   \
                assert(mat1.data[i] != NULL);                          \
                assert(mat2.data[i] != NULL);                          \
                assert(result.data[i] != NULL);                        \
                result.data[i][j] = mat1.data[i][j] + mat2.data[i][j]; \
            }                                                          \
        }                                                              \
    }

/* Multiply two 2D matrices */
#define Matrix2DMult(mat1, mat2, result)                      \
    {                                                         \
        assert(mat1.ncols == mat2.nrows);                     \
        assert(mat1.nrows == result.nrows);                   \
        assert(mat2.ncols == result.ncols);                   \
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
    }

/* Transpose 2D matrix */
#define Matrix2DTrsp(self)                                                             \
    {                                                                                  \
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
    }

/* Free memory allocated to 2D matrix */
#define Matrix2DFree(self)                     \
    {                                          \
        assert(self.data != NULL);             \
        for (int i = 0; i < self.nrows; i++) { \
            assert(self.data[i] != NULL);      \
            free(self.data[i]);                \
        }                                      \
        free(self.data);                       \
        *(&self.data) = NULL;                  \
    }
#endif
