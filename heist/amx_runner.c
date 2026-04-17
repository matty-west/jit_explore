#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    printf("amx_runner: Runner booted. PID: %d\n", getpid());
    printf("amx_runner: Press ENTER to execute cblas_sgemm...\n");
    getchar(); // Blocks here. dyld is completely finished.

    const int N = 8;
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
        C[i] = 0.0f;
    }

    printf("amx_runner: executing cblas_sgemm...\n");
    // Matrix multiplication: C = 1.0 * A * B + 0.0 * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    printf("amx_runner: check C[0] = %f\n", C[0]);
    
    free(A);
    free(B);
    free(C);
    
    printf("amx_runner: done\n");
    return 0;
}
