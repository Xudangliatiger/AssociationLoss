#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void WarpageLossForward(const int nthreads, 
                                        const T* logits,
                                        const int* targets,
                                        const T* iou,
                                        const int num_classes,
                                        const float gamma, 
                                        const float alpha,
                                        const float beta1,
                                        const float beta2,
                                        const int num, 
                                        T* losses)
{
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    int n = i / num_classes;
    int d = i % num_classes; // current class[0~79]; 
    int t = targets[n]; // target class [1~80];

    T x = logits[i];
    T I = iou[n];
    T p = 1. / (1. + expf(-x));

    // Decide it is positive or negative case. 
    int comp = (p<=I);
    T c1 = (comp & t == (d+1));
    T c2 = ((1^comp) & t == (d+1));
    T c3 = (t>=0 & t != (d+1));

    T zn = (1.0 - alpha);

    T logp = logf(max(p, FLT_MIN));
    T log1mp = -1. * x * (x >= 0) - logf(1. + expf(x - 2. * x * (x >= 0)));

    // beta1 * (-I * log(p) + p - I)
    T term1 = beta1 * (-1. * I * logp + p - I);

    // beta2 * ((I-1) * log(1-p) + I - p)
    T term2 = beta2 * ((I - 1.0) * log1mp - p + I);

    // p**gamma * log(1-p)
    T term3 = powf(p, gamma) * log1mp;

    losses[i] = c1 * term1 + c2 * term2 - c3 * term3 * zn;

  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossForward


template <typename T>
__global__ void WarpageLossBackward(const int nthreads,
                                         const T* logits,
                                         const int* targets,
                                         const T* d_losses,
                                         const T* iou,
                                         const int num_classes,
                                         const float gamma,
                                         const float alpha,
                                         const float beta1,
                                         const float beta2,
                                         const int num,
                                         T* d_logits) 
{
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    int n = i / num_classes;
    int d = i % num_classes; // current class[0~79]; 
    int t = targets[n]; // target class [1~80], 0 is background;

    T x = logits[i];
    T I = iou[n];
    T p = 1. / (1. + expf(-x));

    // Decide it is positive or negative case. 
    int comp = (p<=I);
    T c1 = (comp & t == (d+1));
    T c2 = ((1^comp) & t == (d+1));
    T c3 = (t>=0 & t != (d+1));

    T zn = (1.0 - alpha);

    // beta1 * (p - 1) * (I - p)
    T term1 = beta1 * (p - 1.0) * (I - p);

    // beta2 * (-p) * (p - I)
    T term2 = beta2 * p * (I - p);

    // (p**g) * (g*(1-p)*log(1-p) - p)
    T term3 = powf(p, gamma) * ((-1. * x * (x >= 0) - logf(1. + expf(x - 2. * x * (x >= 0)))) * (1. - p) * gamma - p);

    T result = c1 * term1 + c2 * term2 - c3 * term3 * zn;

    d_logits[i] = result * d_losses[i];

  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossBackward


at::Tensor WarpageLoss_forward_cuda(
		const at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& iou,
		const int num_classes, 
		const float gamma, 
		const float alpha,
    const float beta1,
    const float beta2)
{
  AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(iou.type().is_cuda(), "iou must be a CUDA tensor");
  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
	
  auto losses = at::empty({num_samples, logits.size(1)}, logits.options());
  auto losses_size = num_samples * logits.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)losses_size, 512L), 4096L));
  dim3 block(512);

  if (losses.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.type(), "WarpageLoss_forward", [&] {
    WarpageLossForward<scalar_t><<<grid, block, 0, stream>>>(
         losses_size,
         logits.contiguous().data<scalar_t>(),
	       targets.contiguous().data<int>(),
         iou.contiguous().data<scalar_t>(),
         num_classes,
      	 gamma,
      	 alpha,
         beta1,
         beta2,
      	 num_samples,
         losses.data<scalar_t>());
  });

  THCudaCheck(cudaGetLastError());
  return losses;   
}	


at::Tensor WarpageLoss_backward_cuda(
		const at::Tensor& logits,
    const at::Tensor& targets,
		const at::Tensor& d_losses,
    const at::Tensor& iou,
		const int num_classes, 
		const float gamma, 
		const float alpha,
    const float beta1,
    const float beta2)
{
  AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(d_losses.type().is_cuda(), "d_losses must be a CUDA tensor");
  AT_ASSERTM(iou.type().is_cuda(), "iou must be a CUDA tensor");

  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
  AT_ASSERTM(logits.size(1) == num_classes, "logits.size(1) should be num_classes");
	
  auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
  auto d_logits_size = num_samples * logits.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)d_logits_size, 512L), 4096L));
  dim3 block(512);

  if (d_logits.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.type(), "WarpageLoss_backward", [&] {
    WarpageLossBackward<scalar_t><<<grid, block, 0, stream>>>(
         d_logits_size,
         logits.contiguous().data<scalar_t>(),
	       targets.contiguous().data<int>(),
	       d_losses.contiguous().data<scalar_t>(),
         iou.contiguous().data<scalar_t>(),
         num_classes,
      	 gamma,
      	 alpha,
         beta1,
         beta2,
      	 num_samples,
         d_logits.data<scalar_t>());
  });

  THCudaCheck(cudaGetLastError());
  return d_logits;   
}	

