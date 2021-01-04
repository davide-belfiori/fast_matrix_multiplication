#include <iostream>
#include "OpenCLUtils.h"
#include <chrono>

void linear_multiply(float* A, float* B, float* C, int M, int N, int R) {

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {

			// prodotto scalare riga colonna
			float acc = 0.0f;
			for (int r = 0; r < R; r++) {
				acc += A[r*M + m] * B[n*R + r];
			}

			C[n*M + m] = acc;
		}
	}
}

void parallel_multiply(float* A, float* B, float* C, int M, int N, int R) {

	OpenCLUtils utils = OpenCLUtils();

	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	cl::Platform platform = all_Platform[0];

	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);

	cl::Device dev = utils.get_Default_Device();
	cl::Context context({ dev });
	cl::Program program = utils.build_Program_from_Source(dev, context, "kernel_opencl/mat_mul.cl");

	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * M * N);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(float) * N * R);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(float) * M * R);

	cl::CommandQueue queue(context, dev);

	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * M * N, A);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * N * R, B);

	// create the Kernel
	cl::Kernel mat_mul(program, "mat_mul");

	// set arguments
	mat_mul.setArg(0, M);
	mat_mul.setArg(1, N);
	mat_mul.setArg(2, R);
	mat_mul.setArg(3, buffer_A);
	mat_mul.setArg(4, buffer_B);
	mat_mul.setArg(5, buffer_C);

	// run kernel
	queue.enqueueNDRangeKernel(mat_mul, cl::NullRange, cl::NDRange(M, R), cl::NullRange);
	queue.finish();

	//read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * M * R, C);
}

void fill_one(float* M, int M_size) {
	for (int i = 0; i < M_size * M_size; i++) {
		M[i] = 1;
	}
}

void fill_zero(float* M, int M_size) {
	for (int i = 0; i < M_size * M_size; i++) {
		M[i] = 0;
	}
}

int main()
{
	int size[10] = { 10, 20, 40, 100, 200, 400, 600, 800, 1000, 1200 };

	float seq_val[10] = { 0,0,0,0,0,0,0,0,0,0 };
	float par_val[10] = { 0,0,0,0,0,0,0,0,0,0 };

	int num_iter = 1;

	float* A = (float*)malloc(9 * sizeof(float));
	float* C = (float*)malloc(9 * sizeof(float));

	std::cout << "Benchmarking... \n\n";

	for (int k = 0; k < num_iter; k++) {

		for (int i = 0; i < 10; i++) {
			A = (float*)malloc(size[i] * size[i] * sizeof(float));
			C = (float*)malloc(size[i] * size[i] * sizeof(float));

			fill_one(A, size[i]);
			fill_zero(C, size[i]);

			// Sequential Execution

			auto t1 = std::chrono::high_resolution_clock::now();

			linear_multiply(A, A, C, size[i], size[i], size[i]);

			auto t2 = std::chrono::high_resolution_clock::now();
			float exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

			seq_val[i] += exec_time;

			// Parallel Execution

			t1 = std::chrono::high_resolution_clock::now();

			parallel_multiply(A, A, C, size[i], size[i], size[i]);

			t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

			par_val[i] += exec_time;
		}
	}

	std::cout << "Total Iterations = " << num_iter << "\n\n";
	std::cout << "\n Benchmark Result (time expressed in seconds) \n\n";
	std::cout << "***************************************************************************\n\n";
	std::cout << "  Matrix Dimensions --> Sequential Execution Time || Parallel Execution Time \n\n";
	std::cout << "***************************************************************************\n\n";

	for (int i = 0; i < 10; i++) {
		float seq_exec_time = seq_val[i] / num_iter;
		float par_exec_time = par_val[i] / num_iter;

		std::cout << "  (" << size[i] << " x " << size[i] << ") --> " << seq_exec_time;
		std::cout << "  ||  " << par_exec_time << "\n";
	}

}
