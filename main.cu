#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// Dont forget to add cuda_runtime header file
#include <cuda_runtime.h>

constexpr unsigned int N = 5;

__global__ void kernel_expression_evaluation_1(int* ret_01_v, const int* a, const int* b, const int* c, const int* d)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("tid=%d, blockIdx.x=%d, threadIdx.x=%d\n", tid, blockIdx.x, threadIdx.x);
    printf("a[%d]=%d, b[%d]=%d, c[%d]=%d, d[%d]=%d;\n", tid, a[tid], tid,b[tid], tid,c[tid], tid,d[tid]);
    ret_01_v[tid] = (a[tid] + b[tid] * c[tid] > d[tid]);
    // printf("For thread %d, corresponding value is %d\n", tid, ret_01_v[tid]);
}

// __global__ void kernel_scan_x_column(int* ret_x, const int* ret_01_v, const int* x, const int* psum, unsigned int n)
// {
//     const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     const unsigned int stride = blockDim.x * gridDim.x;
//     // unsigned int local_count = psum[tid];
//     for(unsigned int i = tid; i < n; i += stride)
//     {
//         if(ret_01_v[i] == 1)
//         {
//             ret_x[local_count] = x[i];
//         }
//     }
// }

__global__ void kernel_scan_x_column(int* ret_x, const int* ret_01_v, const int* x, const int* psum, unsigned int n)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for(unsigned int i = tid; i < n; i += stride)
    {
        unsigned int local_count = psum[i] - 1;
        if(ret_01_v[i] == 1)
        {
            ret_x[local_count] = x[i];
        }
    }
}

int main() {
    thrust::host_vector<int> a_host(N);
    for (int i = 0; i < N; i++) {
        a_host[i] = i + 1;
    }
    thrust::device_vector<int> a = a_host;

    thrust::host_vector<int> b_host(N, 1);
    thrust::device_vector<int> b = b_host;

    thrust::host_vector<int> c_host(N, 2);
    thrust::device_vector<int> c = c_host;

    thrust::host_vector<int> d_host(N, 3);
    thrust::device_vector<int> d = d_host;

    thrust::host_vector<int> ret_01_v_host(N);
    thrust::device_vector<int> ret_01_v(N);

    // kernel_expression_evaluation_1<<<1, N>>>(ret_01_v.data(), a.data(), b.data(), c.data(), d.data());
    kernel_expression_evaluation_1<<<1, N>>>(thrust::raw_pointer_cast(ret_01_v.data()),
                                             thrust::raw_pointer_cast(a.data()),
                                             thrust::raw_pointer_cast(b.data()),
                                             thrust::raw_pointer_cast(c.data()),
                                             thrust::raw_pointer_cast(d.data()));
    cudaDeviceSynchronize();

    ret_01_v_host = ret_01_v;
    // thrust::copy(ret_01_v.begin(), ret_01_v.end(), ret_01_v_host.begin());
    std::cout << "ret_01_v_host elements are: ";
    for (auto ret_01_v_host_elem: ret_01_v_host) {
        std::cout << ret_01_v_host_elem << " ";
    }
    std::cout << std::endl;

    thrust::device_vector<int> psum(N);
    thrust::inclusive_scan(ret_01_v.begin(), ret_01_v.end(), psum.begin());

    thrust::host_vector<int> psum_host = psum;
    std::cout << "psum_host elements are: ";
    for (auto psum_host_elem: psum_host) {
        std::cout << psum_host_elem << " ";
    }
    std::cout << std::endl;

    thrust::host_vector<int> x_host(N, 10);
    for (int i = 0; i < N; i++) {
        x_host[i] += i + 1;
    }

    thrust::device_vector<int> x = x_host;

    thrust::host_vector<int> ret_x_host;
    thrust::device_vector<int> ret_x(N, 0);
    // thrust::host_vector<int> y_host(N);
    // thrust::host_vector<int> z_host(N);

    kernel_scan_x_column<<<1, 2>>>(thrust::raw_pointer_cast(ret_x.data()),
        thrust::raw_pointer_cast(ret_01_v.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(psum.data()),
        N);

    ret_x_host = ret_x;
    std::cout << "ret_x_host elements are: ";
    for (const auto ret_x_host_elem: ret_x_host) {
        std::cout << ret_x_host_elem << " ";
    }
    std::cout << std::endl;
    return 0;
}