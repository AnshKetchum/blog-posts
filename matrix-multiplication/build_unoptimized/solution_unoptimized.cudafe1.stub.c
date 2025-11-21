#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "solution_unoptimized.fatbin.c"
extern void __device_stub__Z28matrix_multiplication_kernelPKfS0_Pfiii(const float *, const float *, float *, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z28matrix_multiplication_kernelPKfS0_Pfiii(const float *__par0, const float *__par1, float *__par2, int __par3, int __par4, int __par5){__cudaLaunchPrologue(6);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaSetupArgSimple(__par5, 32UL);__cudaLaunch(((char *)((void ( *)(const float *, const float *, float *, int, int, int))matrix_multiplication_kernel)));}
# 12 "src/solution_unoptimized.cu"
void matrix_multiplication_kernel( const float *__cuda_0,const float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 12 "src/solution_unoptimized.cu"
{__device_stub__Z28matrix_multiplication_kernelPKfS0_Pfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 26 "src/solution_unoptimized.cu"
}
# 1 "build_unoptimized/solution_unoptimized.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T5) {  __nv_dummy_param_ref(__T5); __nv_save_fatbinhandle_for_managed_rt(__T5); __cudaRegisterEntry(__T5, ((void ( *)(const float *, const float *, float *, int, int, int))matrix_multiplication_kernel), _Z28matrix_multiplication_kernelPKfS0_Pfiii, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
