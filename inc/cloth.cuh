#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "kernels.cuh"


class Cloth {
public:
    Cloth(int block_size, int num_steps, float y_offset, int num_x, int num_y, float spacing, 
        const Vec3<float>& sphere_center, float sphere_radius);

    ~Cloth();

    // Function to update the mesh
    void update_mesh();

    // Function to reset the cloth
    void reset();

    void simulate_step();

    // Public getters
    int get_num_particles() const { return num_particles; }
    int get_num_tris() const { return num_tris; }
    int get_num_dist_constraints() const { return num_dist_constraints; }
    Vec3<float>* get_device_pos() const { return d_pos; }
    int* get_device_tri_ids() const { return d_tri_ids; }
    Vec3<float>* get_device_normals() const { return d_normals; }
    float* get_device_rest_lengths() const { return d_const_rest_lengths; }

    // Getters for private host arrays
    Vec3<float>* get_host_pos() const { return h_pos; }
    Vec3<float>* get_host_normals() const { return h_normals; }
    float* get_host_inv_mass() const { return h_inv_mass; }
    float* get_host_tri_dist() const { return h_tri_dist; }
    int* get_host_tri_ids() const { return h_tri_ids; }
    Vec3<float>* get_host_corr() const { return h_corr; }

    // function to launch raycast_triangle kernel
    void raycast_triangle_launch(const Vec3<float>& orig, const Vec3<float>& dir);

    // Methods for dragging
    void start_drag(const Vec3<float>& orig, const Vec3<float>& dir);
    void drag(const Vec3<float>& orig, const Vec3<float>& dir);
    void end_drag();

private:

    void init_params(int& num_x, int& num_y);
    void alloc_host_buffers(float y_offset, int num_x, int num_y);
    void alloc_device_buffers();
    void build_constraints(int num_x, int num_y);
    void build_triangles(int num_x, int num_y);

    static constexpr float no_hit = 1.0e6f;

    int block_size;
    int num_particles;
    int num_tris;
    Vec3<float>* d_pos;
    Vec3<float>* d_prev_pos;
    Vec3<float>* d_rest_pos;
    Vec3<float>* d_vel;
    Vec3<float>* d_normals;
    float* d_inv_mass;
    Vec3<float>* d_corr;
    float* d_const_rest_lengths;
    int* d_tri_ids;
    int* d_dist_const_ids;
    float* d_tri_dist;

    Vec3<float> sphere_center;
    float sphere_radius;
    int drag_particle_nr;
    float drag_depth;
    float drag_inv_mass;
    std::vector<int> render_particles;
    float spacing;
    std::vector<int> pass_sizes;
    std::vector<bool> pass_independent;
    int num_dist_constraints;

    float time_step;
    int num_substeps;

    Vec3<float> gravity;
    float jacobi_scale;
    int solve_type;

    // Host arrays
    Vec3<float>* h_pos;
    Vec3<float>* h_normals;
    float* h_inv_mass;
    float* h_tri_dist;
    int* h_tri_ids;
    Vec3<float>* h_corr;
};




