#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "FastNoiseLite.h"  // Updated include (assumes header is in same directory)

namespace py = pybind11;

class Simplex5D {
public:
    Simplex5D(int seed) {
        noise1.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise1.SetFrequency(0.02f);
        noise1.SetSeed(seed);

        noise2.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise2.SetFrequency(0.015f); // Slightly different frequency for variety
        noise2.SetSeed(seed + 1);    // Offset seed for distinct noise
    }

    py::array_t<float> generate_noise(py::array_t<float> t, float x_offset, float y_offset, float z_offset, float w_offset) {
        auto t_buf = t.request();
        float* t_ptr = (float*)t_buf.ptr;
        size_t size = t_buf.shape[0];

        auto result = py::array_t<float>(size);
        auto result_buf = result.request();
        float* result_ptr = (float*)result_buf.ptr;

        for (size_t i = 0; i < size; i++) {
            float t_val = t_ptr[i];
            // 5D simulation: two 3D noise layers with t and w as extra dimensions
            float noise_val1 = noise1.GetNoise(x_offset + t_val, y_offset + t_val, z_offset + t_val);
            float noise_val2 = noise2.GetNoise(x_offset + w_offset, y_offset + w_offset, z_offset + w_offset);
            float noise_val3 = noise1.GetNoise(x_offset + t_val * 1.618f, y_offset + t_val * 1.618f, z_offset + w_offset);  // Blend w into z for 3 args
            result_ptr[i] = 0.5f * noise_val1 + 0.3f * noise_val2 + 0.2f * noise_val3;  // Blend with extra layer
        }

        return result;
    }

private:
    FastNoiseLite noise1;
    FastNoiseLite noise2;
};

PYBIND11_MODULE(simplex5d, m) {
    py::class_<Simplex5D>(m, "Simplex5D")
        .def(py::init<int>())
        .def("generate_noise", &Simplex5D::generate_noise, "Generate 5D Simplex noise",
             py::arg("t"), py::arg("x_offset") = 0.0f, py::arg("y_offset") = 0.0f, 
             py::arg("z_offset") = 0.0f, py::arg("w_offset") = 0.0f);
}