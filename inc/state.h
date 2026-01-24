#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "camera.h"
#include "cloth.cuh"
#include "vec3.cuh"

class State {
public:
    State(int block_size, int argc, char** argv);
    ~State();

    void init();

    void timer_callback(int value);
    void show_screen_cloth();
    void mouse_button(int button, int state, int x, int y);
    void mouse_drag(int x, int y);
    void key_down(unsigned char key, int x, int y);
    void key_up(unsigned char key, int x, int y);
    void process_special_keys(int key, int x, int y);
    void process_special_keys_up(int key, int x, int y);

private:
    void setup_opengl();
    void render();
    std::pair<Vec3<float>, Vec3<float>> get_mouse_ray(int x, int y);
    std::vector<float> convert_vec3_to_float_vector(const Vec3<float>* vec3_array, int num_elements);

    // Init helpers
    void build_ground();

    // Render helpers
    void draw_sphere();

private:
    int screen_width = 900;
    int screen_height = 900;
    int block_size;

    // Ground
    static constexpr int ground_num_tiles = 30;
    static constexpr float ground_tile_size = 0.5f;
    std::vector<float> ground_verts;
    std::vector<float> ground_colors;

    // Simulation
    const float dt = 1.0f / 60.0f;
    std::vector<float> gravity = { 0.0f, -10.0f, 0.0f };
    Vec3<float> sphere_center{ 0.0, 1.5, 0.0 };
    float sphere_radius = 0.4f;

    // Cloth
    Cloth* cloth;
    int cloth_num_x = 200;
    int cloth_num_y = 200;
    float cloth_y = 2.2;
    float cloth_spacing = 0.01;

    // Cloth
    Camera camera;
    bool key_states[256] = { false };

    // Mouse
    bool is_dragging = false;
    bool is_orbiting = false;
    int last_x = 0, last_y = 0;
    glm::dvec3 orbit_center{ 0.0, 0.0, 0.0 };
};

void timer_callback_wrapper(int value);