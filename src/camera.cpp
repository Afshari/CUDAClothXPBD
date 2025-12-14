
#include "camera.h"

Camera::Camera() {
    speed = 0.1;
    right = glm::normalize(glm::cross(forward, up));
    keyDown.assign(256, false);
    specialKeyDown.assign(256, false);
}

void Camera::setView() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, aspect, zNear, zFar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(
        pos.x, pos.y, pos.z,
        pos.x + forward.x, pos.y + forward.y, pos.z + forward.z,
        up.x, up.y, up.z
    );
}

void Camera::lookAt(const glm::dvec3& position, const glm::dvec3& target) {
    pos = position;
    forward = glm::normalize(target - position);
    up = glm::dvec3(0.0, 1.0, 0.0);
    right = glm::normalize(glm::cross(forward, up));
    up = glm::normalize(glm::cross(right, forward));
}

void Camera::handleMouseTranslate(double dx, double dy) {
    double s = glm::length(pos) * 0.001;
    pos -= right * (s * dx);
    pos += up * (s * dy);
}

void Camera::handleWheel(int direction) {
    pos += forward * (direction * speed);
}

void Camera::handleMouseView(double dx, double dy) {
    double s = 0.005;
    glm::dquat qYaw = glm::angleAxis(-dx * s, glm::normalize(up));
    glm::dquat qPitch = glm::angleAxis(-dy * s, glm::normalize(right));

    forward = glm::normalize(qPitch * (qYaw * forward));

    right = glm::normalize(glm::cross(forward, up));
    right.y = 0.0;
    right = glm::normalize(right);
    up = glm::normalize(glm::cross(right, forward));
    forward = glm::normalize(glm::cross(up, right));
}

void Camera::handleKeyDown(unsigned char key, int x, int y) {
    keyDown[key] = true;
}

void Camera::handleKeyUp(unsigned char key, int x, int y) {
    keyDown[key] = false;
}

void Camera::handleSpecialKeyDown(int key, int x, int y) {
    specialKeyDown[key] = true;
}

void Camera::handleSpecialKeyUp(int key, int x, int y) {
    specialKeyDown[key] = false;
}

void Camera::handleSpecialKey() {
    if (specialKeyDown[GLUT_KEY_UP])    pos += forward * speed;
    if (specialKeyDown[GLUT_KEY_DOWN])  pos -= forward * speed;
    if (specialKeyDown[GLUT_KEY_LEFT])  pos -= right * speed;
    if (specialKeyDown[GLUT_KEY_RIGHT]) pos += right * speed;
}

void Camera::handleKeys() {
    if (keyDown['w']) pos += forward * speed;
    if (keyDown['s']) pos -= forward * speed;
    if (keyDown['a']) pos -= right * speed;
    if (keyDown['d']) pos += right * speed;
    if (keyDown['q']) pos += up * speed;
    if (keyDown['e']) pos -= up * speed;
}

void Camera::handleMouseOrbit(double dx, double dy, const glm::dvec3& center) {
    glm::dvec3 offset = pos - center;

    double s = 0.01;
    glm::dquat qYaw = glm::angleAxis(-dx * s, glm::normalize(up));
    glm::dquat qPitch = glm::angleAxis(-dy * s, glm::normalize(right));

    offset = qPitch * (qYaw * offset);
    pos = center + offset;

    forward = glm::normalize(center - pos);
    right = glm::normalize(glm::cross(forward, glm::dvec3(0.0, 1.0, 0.0)));
    up = glm::normalize(glm::cross(right, forward));
}