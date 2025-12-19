# Real-Time Cloth Simulation

**C++ • CUDA • OpenGL**

---

## Table of Contents

- [Real-Time Cloth Simulation](#real-time-cloth-simulation)
  - [Table of Contents](#table-of-contents)
  - [Video](#video)
  - [Key Features](#key-features)
  - [Technologies](#technologies)
  - [Technical Overview](#technical-overview)
  - [Performance Notes](#performance-notes)
  - [Project Structure](#project-structure)
  - [How to Run](#how-to-run)
  - [Background \& Credit](#background--credit)
  - [Future Improvements](#future-improvements)
  - [Author](#author)

---

## Video

[Video](https://www.youtube.com/shorts/XVV7eu9y8m0)

---

## Key Features

* **XPBD-based cloth simulation** for robustness and scalability
* **CUDA parallel computing** for physics solving on the GPU
* **OpenGL rendering** for real-time visualization
* Interactive cloth manipulation via **custom camera & mouse picking**
* Uses **GLM · GLFW · FreeGLUT** for math, windowing, and input

---

## Technologies

* C++14
* CUDA
* OpenGL
* GLM, GLFW, FreeGLUT

---

## Technical Overview

Implements a **Position-Based Dynamics (XPBD) cloth solver fully on the GPU using CUDA**.
Supports both **Jacobi** and **Gauss–Seidel–style** constraint solving through configurable solve passes.
Uses **parallel constraint batching** to maximize GPU occupancy while maintaining stability.
Mouse interaction is handled via **GPU-based triangle raycasting**, enabling interactive cloth dragging.
Collision handling includes **sphere** and **ground plane** constraints.
Surface normals are computed on the GPU for rendering.
Designed with **persistent device buffers** to avoid per-frame allocations and improve performance.

---

## Performance Notes

* Fully GPU-driven physics and constraint solving
* Stable real-time simulation with large cloth resolutions
* Optimized memory usage via persistent CUDA buffers

---

## Project Structure

```text
/src   → source files (C++ & CUDA)
/inc   → header files (H & CUH)
```

---

## How to Run

**Environment:** Visual Studio 2019

**Dependencies (NuGet):**

* glfw 3.4.0
* glm 1.0.2
* glew.v140 1.12.0
* freeglut.3.0.0.v140 1.0.2

Build and run the solution from Visual Studio.

---

## Background & Credit

Inspired by the **“10 Minute Physics”** YouTube channel.
Reimplemented and extended in **C++ / CUDA / OpenGL** with a fully GPU-based pipeline.

---

## Future Improvements

* Support additional soft-body models
* Graph coloring for independent vertex sets to enable more complex parallel simulations

---

## Author

**Mohsen Afshari**
