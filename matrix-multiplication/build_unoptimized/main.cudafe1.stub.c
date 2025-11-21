#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z12flush_kernelPfm(float *, size_t);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z12flush_kernelPfm(float *__par0, size_t __par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(float *, size_t))flush_kernel)));}
# 9 "src/main.cu"
void flush_kernel( float *__cuda_0,size_t __cuda_1)
# 9 "src/main.cu"
{__device_stub__Z12flush_kernelPfm( __cuda_0,__cuda_1);




}
# 1 "build_unoptimized/main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T8) {  __nv_dummy_param_ref(__T8); __nv_save_fatbinhandle_for_managed_rt(__T8); __cudaRegisterEntry(__T8, ((void ( *)(float *, size_t))flush_kernel), _Z12flush_kernelPfm, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
