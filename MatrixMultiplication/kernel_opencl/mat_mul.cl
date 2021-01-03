__kernel void mat_mul(const int M, const int N, const int R,
	const __global float* A,
	const __global float* B,
	__global float* C) {

	// Thread identifiers
	const int globalRow = get_global_id(0);
	const int globalCol = get_global_id(1);

	// Compute a single element (loop over R)
	float acc = 0.0f;
	for (int r = 0; r < R; r++) {
		acc += A[r*M + globalRow] * B[globalCol*R + r];
	}

	// Store the result
	C[globalCol*M + globalRow] = acc;
}
