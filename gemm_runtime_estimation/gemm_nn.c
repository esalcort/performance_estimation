#include <stdlib.h>
#include <stdio.h>

void gemm_nn(int M, int N, int K,
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

float * initialize_array(int size) {
    float *array = (float *) malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        array[i] = ((float)rand()/(float)(RAND_MAX)) * 20 - 10;
    }
    return array;
}

int main( int argc, char *argv[] )  {

    if (argc != 4) {
        printf("Usage: ./gemm <m> <n> <k>\n");
        exit(1);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    float *A = initialize_array( m * k);
    float *B = initialize_array(k * n);
    float *C = (float *) calloc(m * n, sizeof(float));

    gemm_nn(m,n,k,A,k,B,n,C,n);
}