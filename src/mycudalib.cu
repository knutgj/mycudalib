// should work for manual compile
// nvcc --compiler-options '-fPIC' -m 64 -o libmylib.dylib --shared mycudalib.cu

#include <stdio.h> //for io used in hello
#include <cuComplex.h>

#include "mycudalib.h"

const int threadsPerBlock = 256;
  
/* single float element multiplication of A * B into C */
// declare kernel
__global__ void elementMult_kernel(int N, const float* A, const float* B, float* C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] * B[i];
}
void elementMult(int N, const float* A, const float* B, float* C)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementMult_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* destructive single float element multiplication of A * B into B */
// declare kernel
__global__ void elementMultd_kernel(int N, const float* A, float* B)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        B[i] = A[i] * B[i];
}
void elementMultd(int N, const float* A, float* B)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementMultd_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, B);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* complex single float element multiplication of A * B into C */
// declare kernel
__global__ void elementMultc_kernel(int N, const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
      C[i] = cuCmulf(A[i], B[i]);
}
void elementMultc(int N, const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementMultc_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* destructive complex single float element multiplication of A * B into B */
// declare kernel
__global__ void elementMultcd_kernel(int N, const cuFloatComplex* A, cuFloatComplex* B)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
      B[i] = cuCmulf(A[i], B[i]);
}
void elementMultcd(int N, const cuFloatComplex* A, cuFloatComplex* B)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementMultcd_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, B);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* copies data from real-valued A to complex-valued B */ 
// declare kernel
__global__ void cpycomplex_kernel(int N, const float* A, cuFloatComplex* B)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
      B[i] = make_cuFloatComplex(A[i], 0.0);
}
void cpycomplex(int N, const float* A, cuFloatComplex* B)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cpycomplex_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, B);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* copies the real part of complex-valued A to real-valued B */
// declare kernel
__global__ void cpyreal_kernel(int N, const cuFloatComplex* A, float* B)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
      B[i] = cuCrealf(A[i]);
}
void cpyreal(int N, const cuFloatComplex* A, float* B)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cpyreal_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, B);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* sets the ith element of vector A to 0 */
// declare kernel
__global__ void setizero_kernel(float* A, int i)
{
      A[i] = 0.0;
}
void setizero(int N, float* A, int i)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    setizero_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, i);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}

/* sets N elements in A to 1 */
// declare kernel
__global__ void settoone_kernel(int N, float* A)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
      A[i] = 1.0;
}
void settoone(int N, float* A)
{
 // invoke kernel
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    settoone_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A);
    #ifdef _DEBUG
    cudaThreadSynchronize();
    #endif
}


 /* simple hello world to test ok compile and run */
__global__ void hello_k(char *a, int *b) 
{
        a[threadIdx.x] += b[threadIdx.x];
}
int hello()
{
  const int N = 16;
  const int blocksize = 16;
 
  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  char *ad;
  int *bd;
  const int csize = N*sizeof(char);
  const int isize = N*sizeof(int);
  
  printf("%s", a);
  
  cudaMalloc( (void**)&ad, csize ); 
  cudaMalloc( (void**)&bd, isize ); 
  cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
  
  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  hello_k<<<dimGrid, dimBlock>>>(ad, bd);
  cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
  cudaFree( ad );
  
  printf("%s\n", a);
  return EXIT_SUCCESS;
}
