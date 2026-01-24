#include "cloth.cuh"

Cloth::Cloth(int block_size, float y_offset, int num_x, int num_y, float spacing, 
    const Vec3<float>& sphere_center, float sphere_radius)
    : block_size(block_size), sphere_center(sphere_center), sphere_radius(sphere_radius), spacing(spacing) {

    init_params(num_x, num_y);
    alloc_host_buffers(y_offset, num_x, num_y);
    alloc_device_buffers();

    build_constraints(num_x, num_y);
    build_triangles(num_x, num_y);
}

// Destructor
Cloth::~Cloth() {
    // Free host memory
    delete[] h_pos;
    delete[] h_inv_mass;
    delete[] h_corr;
    delete[] h_normals;
    delete[] h_tri_ids;

    // Free device memory
    cudaFree(d_pos);
    cudaFree(d_prev_pos);
    cudaFree(d_rest_pos);
    cudaFree(d_inv_mass);
    cudaFree(d_corr);
    cudaFree(d_vel);
    cudaFree(d_normals);
    cudaFree(d_dist_const_ids);
    cudaFree(d_const_rest_lengths);
    cudaFree(d_tri_ids);
}

void Cloth::init_params(int& num_x, int& num_y) {
    num_substeps = 30;
    time_step = 1.0f / 60.0f;
    solve_type = 0;
    jacobi_scale = 0.2f;
    gravity = Vec3<float>(0.0f, -10.0f, 0.0f);

    drag_particle_nr = -1;
    drag_depth = 0.0f;
    drag_inv_mass = 0.0f;

    if (num_x % 2 == 1) num_x++;
    if (num_y % 2 == 1) num_y++;

    num_particles = (num_x + 1) * (num_y + 1);
}

void Cloth::alloc_host_buffers(float y_offset, int num_x, int num_y) {
    h_pos = new Vec3<float>[num_particles];
    h_normals = new Vec3<float>[num_particles];
    h_inv_mass = new float[num_particles];
    h_corr = new Vec3<float>[num_particles];

    for (int i = 0; i < num_particles; i++) {
        h_inv_mass[i] = 1.0f;
        h_corr[i] = Vec3<float>(0.0f, 0.0f, 0.0f);
        h_normals[i] = Vec3<float>(0.0f, 0.0f, 0.0f);
    }

    // initialize positions
    for (int xi = 0; xi <= num_x; ++xi) {
        for (int yi = 0; yi <= num_y; ++yi) {
            int id = xi * (num_y + 1) + yi;
            h_pos[id] = Vec3<float>(
                ((-num_x * 0.5f + xi) * spacing),
                y_offset,
                ((-num_y * 0.5f + yi) * spacing)
                );
        }
    }
}

void Cloth::alloc_device_buffers() {
    cudaMalloc(&d_pos, num_particles * sizeof(Vec3<float>));
    cudaMalloc(&d_prev_pos, num_particles * sizeof(Vec3<float>));
    cudaMalloc(&d_rest_pos, num_particles * sizeof(Vec3<float>));
    cudaMalloc(&d_inv_mass, num_particles * sizeof(float));
    cudaMalloc(&d_corr, num_particles * sizeof(Vec3<float>));
    cudaMalloc(&d_vel, num_particles * sizeof(Vec3<float>));
    cudaMalloc(&d_normals, num_particles * sizeof(Vec3<float>));

    cudaMemcpy(d_corr, h_corr, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_pos, h_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rest_pos, h_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_mass, h_inv_mass, num_particles * sizeof(float), cudaMemcpyHostToDevice);
}

void Cloth::build_constraints(int num_x, int num_y) {
    pass_sizes = {
        (num_x + 1) * int(floor(num_y / 2)),
        (num_x + 1) * int(floor(num_y / 2)),
        int(floor(num_x / 2)) * (num_y + 1),
        int(floor(num_x / 2)) * (num_y + 1),
        2 * num_x * num_y + (num_x + 1) * (num_y - 1) + (num_y + 1) * (num_x - 1)
    };

    pass_independent = { true, true, true, true, false };

    num_dist_constraints = 0;
    for (int pass_size : pass_sizes) {
        num_dist_constraints += pass_size;
    }

    std::vector<int> dist_const_ids(2 * num_dist_constraints);

    int i = 0;

    for (int pass_nr = 0; pass_nr < 2; ++pass_nr) {
        for (int xi = 0; xi <= num_x; ++xi) {
            for (int yi = 0; yi < num_y / 2; ++yi) {
                dist_const_ids[2 * i] = xi * (num_y + 1) + 2 * yi + pass_nr;
                dist_const_ids[2 * i + 1] = xi * (num_y + 1) + 2 * yi + pass_nr + 1;
                ++i;
            }
        }
    }

    for (int pass_nr = 0; pass_nr < 2; ++pass_nr) {
        for (int xi = 0; xi < num_x / 2; ++xi) {
            for (int yi = 0; yi <= num_y; ++yi) {
                dist_const_ids[2 * i] = (2 * xi + pass_nr) * (num_y + 1) + yi;
                dist_const_ids[2 * i + 1] = (2 * xi + pass_nr + 1) * (num_y + 1) + yi;
                ++i;
            }
        }
    }

    for (int xi = 0; xi < num_x; ++xi) {
        for (int yi = 0; yi < num_y; ++yi) {
            dist_const_ids[2 * i] = xi * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = (xi + 1) * (num_y + 1) + yi + 1;
            ++i;

            dist_const_ids[2 * i] = (xi + 1) * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = xi * (num_y + 1) + yi + 1;
            ++i;
        }
    }

    for (int xi = 0; xi <= num_x; ++xi) {
        for (int yi = 0; yi < num_y - 1; ++yi) {
            dist_const_ids[2 * i] = xi * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = xi * (num_y + 1) + yi + 2;
            ++i;
        }
    }

    for (int xi = 0; xi < num_x - 1; ++xi) {
        for (int yi = 0; yi <= num_y; ++yi) {
            dist_const_ids[2 * i] = xi * (num_y + 1) + yi;
            dist_const_ids[2 * i + 1] = (xi + 2) * (num_y + 1) + yi;
            ++i;
        }
    }

    cudaMalloc(&d_dist_const_ids, 2 * num_dist_constraints * sizeof(int));
    cudaMemcpy(d_dist_const_ids, dist_const_ids.data(),
        2 * num_dist_constraints * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_const_rest_lengths, num_dist_constraints * sizeof(float));
    cudaMemset(d_const_rest_lengths, 0, num_dist_constraints * sizeof(float));

    int threads_per_block = block_size;
    int blocks_per_grid = (num_dist_constraints + threads_per_block - 1) / threads_per_block;
    compute_rest_lengths << <blocks_per_grid, threads_per_block >> > (
        d_pos, d_dist_const_ids, d_const_rest_lengths, num_dist_constraints
        );
    cudaDeviceSynchronize();
}

void Cloth::build_triangles(int num_x, int num_y) {
    num_tris = 2 * num_x * num_y;

    h_tri_ids = new int[3 * num_tris];
    for (int i = 0; i < 3 * num_tris; ++i) {
        h_tri_ids[i] = 0;
    }

    int i = 0;
    for (int xi = 0; xi < num_x; ++xi) {
        for (int yi = 0; yi < num_y; ++yi) {
            int id0 = xi * (num_y + 1) + yi;
            int id1 = (xi + 1) * (num_y + 1) + yi;
            int id2 = (xi + 1) * (num_y + 1) + yi + 1;
            int id3 = xi * (num_y + 1) + yi + 1;

            h_tri_ids[i] = id0;
            h_tri_ids[i + 1] = id1;
            h_tri_ids[i + 2] = id2;

            h_tri_ids[i + 3] = id0;
            h_tri_ids[i + 4] = id2;
            h_tri_ids[i + 5] = id3;

            i += 6;
        }
    }

    cudaMalloc(&d_tri_ids, 3 * num_tris * sizeof(int));
    cudaMemcpy(d_tri_ids, h_tri_ids, 3 * num_tris * sizeof(int), cudaMemcpyHostToDevice);

    h_tri_dist = new float[num_tris];
    std::fill(h_tri_dist, h_tri_dist + num_tris, no_hit);

    cudaMalloc(&d_tri_dist, num_tris * sizeof(float));
    cudaMemcpy(d_tri_dist, h_tri_dist, num_tris * sizeof(float), cudaMemcpyHostToDevice);
}


void Cloth::simulate_step() {
    float dt = time_step / num_substeps;
    int num_passes = pass_sizes.size();

    for (int step = 0; step < num_substeps; ++step) {
        // Step 1: Integrate positions and velocities
        integrate << <(num_particles + block_size-1) / block_size, block_size >> > (dt, gravity, d_inv_mass,
            d_prev_pos, d_pos, d_vel, sphere_center, sphere_radius, num_particles);
        cudaDeviceSynchronize();

        // Step 2: Solve constraints
        if (solve_type == 0) {
            int first_constraint = 0;
            for (int pass_nr = 0; pass_nr < num_passes; ++pass_nr) {
                int num_constraints = pass_sizes[pass_nr];

                if (pass_independent[pass_nr]) {
                    solve_distance_constraints << <(num_constraints + block_size-1) / block_size, block_size >> > (0, first_constraint, d_inv_mass, d_pos, d_corr,
                        d_dist_const_ids, d_const_rest_lengths, num_constraints);
                    cudaDeviceSynchronize();
                }
                else {
                    cudaMemset(d_corr, 0, num_particles * sizeof(Vec3<float>));
                    solve_distance_constraints << <(num_constraints + block_size-1) / block_size, block_size >> > (1, first_constraint, d_inv_mass, d_pos, d_corr,
                        d_dist_const_ids, d_const_rest_lengths, num_constraints);
                    cudaDeviceSynchronize();
                    add_corrections << <(num_particles + block_size-1) / block_size, block_size >> > (d_pos, d_corr, jacobi_scale, num_particles);
                    cudaDeviceSynchronize();
                }
                first_constraint += num_constraints;
            }
        }
        else if (solve_type == 1) {
            cudaMemset(d_corr, 0, num_particles * sizeof(Vec3<float>));
            solve_distance_constraints << <(num_dist_constraints + block_size-1) / block_size, block_size >> > (1, 0, d_inv_mass, d_pos, d_corr,
                d_dist_const_ids, d_const_rest_lengths, num_dist_constraints);
            cudaDeviceSynchronize();
            add_corrections << <(num_particles + block_size-1) / block_size, block_size >> > (d_pos, d_corr, jacobi_scale, num_particles);
            cudaDeviceSynchronize();
        }

        // Step 3: Update velocities based on the new positions
        update_vel << <(num_particles + block_size-1) / block_size, block_size >> > (dt, d_prev_pos, d_pos, d_vel, num_particles);
        cudaDeviceSynchronize();
    }

    // Step 4: Copy the updated positions back to the host
    cudaMemcpy(h_pos, d_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyDeviceToHost);
}

void Cloth::update_mesh() {
    // Set the normals array to zero
    cudaMemset(d_normals, 0, num_particles * sizeof(Vec3<float>));

    int threads_per_block = block_size;
    int blocks_per_grid = (num_tris + threads_per_block - 1) / threads_per_block;
    add_normals << <blocks_per_grid, threads_per_block >> > (d_pos, d_tri_ids, d_normals, num_tris);
    cudaDeviceSynchronize();

    blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;
    normalize_normals << <blocks_per_grid, threads_per_block >> > (d_normals, num_particles);
    cudaDeviceSynchronize();

    cudaMemcpy(h_normals, d_normals, num_particles * sizeof(Vec3<float>), cudaMemcpyDeviceToHost);
}

void Cloth::reset() {
    // Set velocity array to zero
    cudaMemset(d_vel, 0, num_particles * sizeof(Vec3<float>));

    // Copy rest positions to current positions
    cudaMemcpy(d_pos, d_rest_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyDeviceToDevice);
}

void Cloth::raycast_triangle_launch(const Vec3<float>& orig, const Vec3<float>& dir) {
    // initialize distances on device (no_hit)
    std::fill(h_tri_dist, h_tri_dist + num_tris, no_hit);
    cudaMemcpy(d_tri_dist, h_tri_dist, num_tris * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = block_size;
    int blocks_per_grid = (num_tris + threads_per_block - 1) / threads_per_block;

    raycast_triangle << <blocks_per_grid, threads_per_block >> > (
        orig, dir, d_pos, d_tri_ids, d_tri_dist, num_tris);
    cudaDeviceSynchronize();

    cudaMemcpy(h_tri_dist, d_tri_dist, num_tris * sizeof(float), cudaMemcpyDeviceToHost);
}


void Cloth::start_drag(const Vec3<float>& orig, const Vec3<float>& dir) {

    // Launch the raycast_triangle kernel
    raycast_triangle_launch(orig, dir);

    // Initialize drag_depth
    drag_depth = 0.0f;

    // Find the triangle with the minimum distance to the ray
    int min_tri_nr = std::min_element(h_tri_dist, h_tri_dist + num_tris) - h_tri_dist;

    // Check if the closest triangle was hit
    if (h_tri_dist[min_tri_nr] < 1.0e6) {
        // Identify the particle associated with the closest triangle
        drag_particle_nr = h_tri_ids[3 * min_tri_nr];
        drag_depth = h_tri_dist[min_tri_nr];
        drag_inv_mass = h_inv_mass[drag_particle_nr];

        // Lock the particle by setting its inverse mass to 0
        h_inv_mass[drag_particle_nr] = 0.0f;
        cudaMemcpy(d_inv_mass, h_inv_mass, num_particles * sizeof(float), cudaMemcpyHostToDevice);

        // Calculate the drag position based on the ray origin and direction
        Vec3<float> drag_pos = orig + dir * drag_depth;
        h_pos[drag_particle_nr] = drag_pos;

        // Copy the updated particle position back to the device
        cudaMemcpy(d_pos, h_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice);
    }
}

void Cloth::drag(const Vec3<float>& orig, const Vec3<float>& dir) {
    if (drag_particle_nr >= 0) {
        Vec3<float> drag_pos = orig + dir * drag_depth;
        h_pos[drag_particle_nr] = drag_pos;
        cudaMemcpy(d_pos, h_pos, num_particles * sizeof(Vec3<float>), cudaMemcpyHostToDevice);
    }
}

void Cloth::end_drag() {
    if (drag_particle_nr >= 0) {
        h_inv_mass[drag_particle_nr] = drag_inv_mass;
        cudaMemcpy(d_inv_mass, h_inv_mass, num_particles * sizeof(float), cudaMemcpyHostToDevice);
        drag_particle_nr = -1;
    }
}


