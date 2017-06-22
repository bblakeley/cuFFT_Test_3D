NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall --gpu-architecture=sm_30
CUFFTLIB = -L/usr/local/cuda/lib -lcufft -I/usr/local/cuda/inc

main.exe: cuFFT_Test_3D.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUFFTLIB)

clean:
	rm -f *.o *.exe
