
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <vector>
#include <cmath>

#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>

#include "state.h"

State* g_state = nullptr;

void timer_callback_wrapper(int value) {
    g_state->timer_callback(value);
}
void show_screen_cloth_wrapper() {
    g_state->show_screen_cloth();
}
void mouse_button_wrapper(int button, int state, int x, int y) {
    g_state->mouse_button(button, state, x, y);
}
void mouse_drag_wrapper(int x, int y) {
    g_state->mouse_drag(x, y);
}
void key_down_wrapper(unsigned char key, int x, int y) {
    g_state->key_down(key, x, y);
}
void key_up_wrapper(unsigned char key, int x, int y) {
    g_state->key_up(key, x, y);
}
void process_special_keys_wrapper(int key, int x, int y) {
    g_state->process_special_keys(key, x, y);
}
void process_special_keys_up_wrapper(int key, int x, int y) {
    g_state->process_special_keys_up(key, x, y);
}


int main(int argc, char** argv) {

    std::cout << "------------- START OF EXECUTION -------------" << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(2));

    State state(argc, argv);
    g_state = &state;

    state.init();

    glutDisplayFunc(show_screen_cloth_wrapper);
    glutMouseFunc(mouse_button_wrapper);
    glutMotionFunc(mouse_drag_wrapper);
    glutKeyboardFunc(key_down_wrapper);
    glutKeyboardUpFunc(key_up_wrapper);
    glutSpecialFunc(process_special_keys_wrapper);
    glutSpecialUpFunc(process_special_keys_up_wrapper);
    glutTimerFunc(0, timer_callback_wrapper, 0);

    glutMainLoop();


    return 0;
}
