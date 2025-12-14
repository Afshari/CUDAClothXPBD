#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>


class Camera {
public:
    Camera();

    void setView();
    void lookAt(const glm::dvec3& position, const glm::dvec3& target);

    void handleMouseTranslate(double dx, double dy);
    void handleWheel(int direction);
    void handleMouseView(double dx, double dy);
    void handleKeyDown(unsigned char key, int x, int y);
    void handleKeyUp(unsigned char key, int x, int y);
    void handleSpecialKeyDown(int key, int x, int y);
    void handleSpecialKeyUp(int key, int x, int y);
    void handleSpecialKey();
    void handleKeys();
    void handleMouseOrbit(double dx, double dy, const glm::dvec3& center);

private:
    glm::dvec3 pos{ 0.0, 1.0, 5.0 };
    glm::dvec3 forward{ 0.0, 0.0, -1.0 };
    glm::dvec3 up{ 0.0, 1.0, 0.0 };
    glm::dvec3 right{ 1.0, 0.0, 0.0 };

    double fov = 60.0;   // degrees
    double aspect = 1.0;  // width / height
    double zNear = 0.1;
    double zFar = 1000.0;

    double speed;
    std::vector<bool> keyDown;
    std::vector<bool> specialKeyDown;

};
