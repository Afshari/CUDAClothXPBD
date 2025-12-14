
#include "kernels.cuh"
#include "vec3.cuh"
#include <cassert>

__device__ float length(const Vec3<float>& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__global__ void compute_rest_lengths(const Vec3<float>* pos, const int* const_ids, float* rest_lengths, int num_constraints) {
    int c_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_nr >= num_constraints) return;

    Vec3<float> p0 = pos[const_ids[2 * c_nr]];
    Vec3<float> p1 = pos[const_ids[2 * c_nr + 1]];
    rest_lengths[c_nr] = (p1 - p0).length();
}


__global__ void integrate(
    float dt, Vec3<float> gravity, float* inv_mass, Vec3<float>* prev_pos,
    Vec3<float>* pos, Vec3<float>* vel, Vec3<float> sphere_center, float sphere_radius, int num_particles) {

    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr >= num_particles) return;

    prev_pos[p_nr] = pos[p_nr];
    if (inv_mass[p_nr] == 0.0f) {
        return;
    }
    vel[p_nr] = vel[p_nr] + gravity * dt;
    pos[p_nr] = pos[p_nr] + vel[p_nr] * dt;

    // collisions
    float thickness = 0.001f;
    float friction = 0.01f;

    float d = length(pos[p_nr] - sphere_center);
    if (d < (sphere_radius + thickness)) {
        Vec3<float> p = pos[p_nr] * (1.0f - friction) + prev_pos[p_nr] * friction;
        Vec3<float> r = p - sphere_center;
        d = length(r);
        pos[p_nr] = sphere_center + r * ((sphere_radius + thickness) / d);
    }

    Vec3<float> p = pos[p_nr];
    if (p.y < thickness) {
        p = pos[p_nr] * (1.0f - friction) + prev_pos[p_nr] * friction;
        pos[p_nr] = Vec3<float>(p.x, thickness, p.z);
    }
}


__global__ void solve_distance_constraints(
    int solve_type, int first_constraint, float* inv_mass, Vec3<float>* pos,
    Vec3<float>* corr, int* const_ids, float* rest_lengths, int num_constraints) {

    int cNr = first_constraint + blockIdx.x * blockDim.x + threadIdx.x;
    if (cNr >= first_constraint + num_constraints) return;

    int id0 = const_ids[2 * cNr];
    int id1 = const_ids[2 * cNr + 1];
    float w0 = inv_mass[id0];
    float w1 = inv_mass[id1];
    float w = w0 + w1;
    if (w == 0.0f) {
        return;
    }
    Vec3<float> p0 = pos[id0];
    Vec3<float> p1 = pos[id1];
    Vec3<float> d = p1 - p0;
    Vec3<float> n = d.normalized();
    float l = length(d);
    float l0 = rest_lengths[cNr];
    Vec3<float> dP = n * (l - l0) / w;
    if (solve_type == 1) {
        atomic_add(&corr[id0], dP * w0);
        atomic_sub(&corr[id1], dP * w1);
    }
    else {
        atomic_add(&pos[id0], dP * w0);
        atomic_sub(&pos[id1], dP * w1);
    }
}


__global__ void add_normals(
    Vec3<float>* pos, int* tri_ids, Vec3<float>* normals, int num_triangles) {

    int tri_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_nr >= num_triangles) return;

    int id0 = tri_ids[3 * tri_nr];
    int id1 = tri_ids[3 * tri_nr + 1];
    int id2 = tri_ids[3 * tri_nr + 2];

    Vec3<float> a = pos[id1] - pos[id0];
    Vec3<float> b = pos[id2] - pos[id0];
    Vec3<float> normal = cross(a, b);

    atomic_add(&normals[id0], normal);
    atomic_add(&normals[id1], normal);
    atomic_add(&normals[id2], normal);
}


__global__ void normalize_normals(Vec3<float>* normals, int num_normals) {
    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr < num_normals) {
        normals[p_nr] = normals[p_nr].normalized();
    }
}

__global__ void add_corrections(Vec3<float>* pos, Vec3<float>* corr, float scale, int num_particles) {
    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr >= num_particles) return;

    pos[p_nr] = pos[p_nr] + corr[p_nr] * scale;
}

__global__ void update_vel(float dt, Vec3<float>* prev_pos, Vec3<float>* pos, Vec3<float>* vel, int num_particles) {
    int p_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_nr >= num_particles) return;

    vel[p_nr] = (pos[p_nr] - prev_pos[p_nr]) / dt;
}


void update_mesh(Vec3<float>* d_pos, int* d_tri_ids, Vec3<float>* d_normals, Vec3<float>* h_normals,
    int num_tris, int num_particles) {

    // Zero out the normals on the device
    cudaMemset(d_normals, 0, num_particles * sizeof(Vec3<float>));

    // Launch the add_normals kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_tris + threadsPerBlock - 1) / threadsPerBlock;
    add_normals << <blocksPerGrid, threadsPerBlock >> > (d_pos, d_tri_ids, d_normals, num_tris);
    cudaDeviceSynchronize();

    // Launch the normalize_normals kernel
    blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;
    normalize_normals << <blocksPerGrid, threadsPerBlock >> > (d_normals, num_particles);
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(h_normals, d_normals, num_particles * sizeof(Vec3<float>), cudaMemcpyDeviceToHost);
}


__global__ void raycast_triangle(
    Vec3<float> orig, Vec3<float> dir, Vec3<float>* pos,
    int* tri_ids, float* dist, int num_tris) {

    int tri_nr = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_nr >= num_tris) return;

    float no_hit = 1.0e6;

    int id0 = tri_ids[3 * tri_nr];
    int id1 = tri_ids[3 * tri_nr + 1];
    int id2 = tri_ids[3 * tri_nr + 2];

    Vec3<float> edge1 = pos[id1] - pos[id0];
    Vec3<float> edge2 = pos[id2] - pos[id0];
    Vec3<float> pvec = cross(dir, edge2);
    float det = dot(edge1, pvec);

    if (det == 0.0f) {
        dist[tri_nr] = no_hit;
        return;
    }

    float inv_det = 1.0f / det;
    Vec3<float> tvec = orig - pos[id0];
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        dist[tri_nr] = no_hit;
        return;
    }

    Vec3<float> qvec = cross(tvec, edge1);
    float v = dot(dir, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        dist[tri_nr] = no_hit;
        return;
    }

    dist[tri_nr] = dot(edge2, qvec) * inv_det;
}



#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

