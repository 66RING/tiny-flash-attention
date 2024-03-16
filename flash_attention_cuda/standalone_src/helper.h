#pragma once

bool all_close(float *A, float *B, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    if (fabs(A[i] - B[i]) > 1e-5) {
      printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
      return false;
    }
  }
  return true;
}

// print matrix
void print_host_matrix(float *matrix, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", matrix[i * n + j]);
    }
    printf("\n");
  }
}

void print_device_matrix(float *dev_ptr, int m, int n) {
  float *host_ptr = new float[m * n];
  cudaMemcpy(host_ptr, dev_ptr, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", host_ptr[i * n + j]);
    }
    printf("\n");
  }
  free(host_ptr);
}

