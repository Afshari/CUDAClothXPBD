
#include "vec3.cuh"

__device__ float dot(const Vec3<float>& a, const Vec3<float>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3<float> cross(const Vec3<float>& a, const Vec3<float>& b) {
    return Vec3<float>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
        );
}

__global__ void accumulate(float* array, int* indices, float* values, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        atomicAdd(&array[indices[idx]], values[idx]);
    }
}

__device__ void atomic_add(Vec3<float>* address, const Vec3<float>& value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
    atomicAdd(&(address->z), value.z);
}

__device__ void atomic_sub(Vec3<float>* address, const Vec3<float>& value) {
    atomicAdd(&(address->x), -value.x);
    atomicAdd(&(address->y), -value.y);
    atomicAdd(&(address->z), -value.z);
}

