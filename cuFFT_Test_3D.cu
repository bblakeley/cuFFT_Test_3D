/* This program is designed to take the first
 derivative of a 3 dimensional function */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes, project
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define L (2.0*M_PI)
#define TX 8
#define TY 8
#define TZ 8

int divUp(int a, int b) { return (a + b - 1) / b; }

void writeData(const char *name, double *data, int size)
{
	int i = 0;
	FILE *out = fopen(name, "wb");
	for (i = 0; i < size; ++i){
		fwrite( (void*)(&data[i]), sizeof(data[i]), 1, out);
	}

	fclose(out);

	return;
}

__device__
int idxClip(int idx, int idxMax){
	return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int stack, int width, int height, int depth){
	return idxClip(stack, depth) + idxClip(row, height)*depth + idxClip(col, width)*depth*height;
	// Note: using column-major indexing format
}

__global__ 
void initialize(cufftDoubleReal *f1, cufftDoubleReal *f2)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, NZ);

	// Create physical vectors in temporary memory
	double x = i * (double)L / NX;
	double y = j * (double)L / NY;
	double z = k * (double)L / NZ;

	// Initialize starting array
	f1[idx] = sin(x)*cos(y)*cos(z);

	// Calculate exact derivative of starting array
	f2[idx] = cos(x)*cos(y)*cos(z);

	return;
}

__global__
void waveDomain(double *k1)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX/2)
		k1[i] = (double)i;
	else
		k1[i] = (double)i - NX;

	return;
}

__global__
void multIk(cufftDoubleComplex *f, cufftDoubleComplex *fIk, double *wave, const int dir)
{	// Function to multiply the function fhat by i*k
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || (j >= NY) || (k >= NZ2)) return;
	const int idx = flatten(i, j, k, NX, NY, NZ2);

	// i*k*(a + bi) = -k*b + i*k*a

	if(dir == 1){ // Takes derivative in 1 direction (usually x)
		fIk[idx].x = -wave[i]*f[idx].y;
		fIk[idx].y = wave[i]*f[idx].x;
	}
	if(dir == 2){	// Takes derivative in 2 direction (usually y)
		fIk[idx].x = -wave[j]*f[idx].y;
		fIk[idx].y = wave[j]*f[idx].x;
	}
	if(dir == 3){
		fIk[idx].x = -wave[k]*f[idx].y;
		fIk[idx].y = wave[k]*f[idx].x;
	}

	return;
}

__global__
void complexScale( cufftDoubleComplex *f)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || (j >= NY) || (k >= NZ2)) return;
	const int idx = flatten(i, j, k, NX, NY, NZ2);

	f[idx].x = f[idx].x / ( (double)NN );
	f[idx].y = f[idx].y / ( (double)NN );

	return;
}

void fftder(cufftHandle p, cufftHandle invp, double *wave, cufftDoubleReal *f, cufftDoubleReal *fp, int dir)
{
	// Define loop variables
	cufftDoubleComplex *fhat;
	cufftDoubleComplex *fphat;

	// Allocate memory for temporary arrays
	cudaMallocManaged((void**)&fhat, sizeof(cufftDoubleComplex)*NX*NY*(NZ2) );
	cudaMallocManaged((void**)&fphat, sizeof(cufftDoubleComplex)*NX*NY*(NZ2) );

	// Take Fourier Transform of function
	if (cufftExecD2Z(p, f, fhat ) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecD2Z failed");
		return;	
	}

	// Launch CUDA kernel to take derivative in Fourier Space;
	const dim3 blockSize(TX, TY, TZ);
	const dim3 gridSize(divUp(NX, TX), divUp(NY, TY), divUp(NZ, TZ));
	// Multiplies fhat by I*k (definition of derivative in wavespace)
	multIk<<<gridSize, blockSize>>>(fhat, fphat, wave, dir);
	// Scales the vectors by NX*NY*NZ to account for Fourier Transform
	complexScale<<<gridSize, blockSize>>>(fphat);

	// Inverse Fourier Transform to physical space.
	if (cufftExecZ2D(invp, fphat, fp) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecZ2D failed");
	return;	}

	// Clear memory
	cudaFree(fhat);
	cudaFree(fphat);

	return;
}

int main (void)
{
	// Set CUDA to run on GTX 980 (currently device 0)
	int deviceNum = 0;

	cudaSetDevice(deviceNum);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceNum);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
 			2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

	//Create plan for cuFFT
	cufftHandle plan;
	cufftHandle invplan;

	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecD2Z failed");
		cudaError_t err = cudaGetLastError(); }

	if (cufftPlan3d(&invplan, NX, NY, NZ, CUFFT_Z2D)  != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecD2Z failed");
		cudaError_t err = cudaGetLastError(); }

	// Declare variables
	int i, j, k, idx;
	double *kx = 0;
	cufftDoubleReal *u;
	cufftDoubleReal *dudx;
	cufftDoubleReal *dudx_exact;

	// Allocate memory for arrays
	cudaMallocManaged(&kx, NX*sizeof(double));
	cudaMallocManaged(&u, sizeof(cufftDoubleComplex)*NX*NY*NZ2 );
	cudaMallocManaged(&dudx, sizeof(cufftDoubleReal)*NN );
	cudaMallocManaged(&dudx_exact, sizeof(cufftDoubleReal)*NN );

	// Launch CUDA kernel to initialize velocity field
	const dim3 blockSize(TX, TY, TZ);
	const dim3 gridSize(divUp(NX, TX), divUp(NY, TY), divUp(NZ, TZ));
	
	initialize<<<gridSize, blockSize>>>(u, dudx_exact);
	
	// Setup wavespace domain
	waveDomain<<<gridSize, blockSize>>>(kx);

	cudaDeviceSynchronize();		// Added this line to prevent compiler error: "Bus error (core dumped)".

	// Take derivative using Fourier Transforms
	fftder(plan, invplan, kx, u, dudx, 1);

	// takeFFT(plan, invplan, u, u_fft);
	cudaDeviceSynchronize();

	printf("Using the GPU %s. The number of elements in the array is %2d.\n",prop.name,NX*NY*NZ);
	printf("Displaying 50 results:\n");
	printf("  Index         Error     \n");
	printf(" -------     ----------- \n");
	for (i = 0; i < 1; ++i){
		for (j = 0; j < 1; ++j){
			for (k = 50; k < 101; ++k){
				idx = k + NZ*j + NZ*NY*i;
				printf(" %d             %4.4g \n",idx, dudx[idx]-dudx_exact[idx]);
			}
		}
	}

	// Write output data to file (for additional analysis if desired)
	// writeData("dudx",dudx, NX*NY*NZ);
	// writeData("dudx_exact", dudx_exact, NX*NY*NZ);
	// writeData("u", u, NX*NY*NZ);

	//Clean variables
	cufftDestroy(plan);
	cufftDestroy(invplan);

	cudaFree(u);
	cudaFree(dudx);
	cudaFree(dudx_exact);
	cudaFree(kx);

	cudaDeviceReset();

	return 0;
}
