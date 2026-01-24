#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "vec3.cuh"

__device__ float length(const Vec3<float>& v);

__global__ void compute_rest_lengths(const Vec3<float>* pos, const int* constIds, float* restLengths, int numConstraints);
__global__ void integrate(float dt, Vec3<float> gravity, float* invMass, Vec3<float>* prevPos,
    Vec3<float>* pos, Vec3<float>* vel, Vec3<float> sphereCenter,
    float sphereRadius, int numParticles);
__global__ void solve_distance_constraints(int solveType, int firstConstraint, float* invMass,
    Vec3<float>* pos, Vec3<float>* corr, int* constIds,
    float* restLengths, int numConstraints);
__global__ void add_corrections(Vec3<float>* pos, Vec3<float>* corr, float scale, int numParticles);
__global__ void update_vel(float dt, Vec3<float>* prevPos, Vec3<float>* pos, Vec3<float>* vel, int numParticles);
__global__ void add_normals(Vec3<float>* pos, int* triIds, Vec3<float>* normals, int numTriangles);
__global__ void normalize_normals(Vec3<float>* normals, int numNormals);

__global__ void raycast_triangle(Vec3<float> orig, Vec3<float> dir, Vec3<float>* pos, int* tri_ids, float* dist, int num_tris);

