#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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

    py::array_t<float> generate_noise(py::array_t<float> t_array, float x_offset, float y_offset, float z_offset, float w_offset) {
        // Obtain unchecked accessor and allocate result BEFORE releasing GIL
        auto t = t_array.unchecked<1>();
        size_t size = t.shape(0);

        auto result = py::array_t<float>(size);
        auto r = result.mutable_unchecked<1>();

        {
            py::gil_scoped_release release;  // Release GIL for computation

            for (size_t i = 0; i < size; i++) {
                float t_val = t(i);
                // 5D simulation: two 3D noise layers with t and w as extra dimensions
                float noise_val1 = noise1.GetNoise(x_offset + t_val, y_offset + t_val, z_offset + t_val);
                float noise_val2 = noise2.GetNoise(x_offset + w_offset, y_offset + w_offset, z_offset + w_offset);
                float noise_val3 = noise1.GetNoise(x_offset + t_val * 1.618f, y_offset + t_val * 1.618f, z_offset + w_offset);  // Blend w into z for 3 args
                r(i) = 0.5f * noise_val1 + 0.3f * noise_val2 + 0.2f * noise_val3;  // Blend with extra layer
            }
        }

        return result;
    }

    py::list batch_generate(py::array_t<float> t_array, py::list params) {
        // Obtain unchecked accessor BEFORE releasing GIL
        auto t = t_array.unchecked<1>();
        size_t size = t.shape(0);
        size_t num_calls = params.size();

        // Parse all parameter tuples and allocate result arrays while holding GIL
        struct NoiseParams {
            float y, z, w, v;
        };
        std::vector<NoiseParams> param_vec(num_calls);
        std::vector<py::array_t<float>> results(num_calls);
        std::vector<float*> result_ptrs(num_calls);

        for (size_t j = 0; j < num_calls; j++) {
            py::tuple tup = params[j].cast<py::tuple>();
            param_vec[j].y = tup[0].cast<float>();
            param_vec[j].z = tup[1].cast<float>();
            param_vec[j].w = tup[2].cast<float>();
            param_vec[j].v = tup[3].cast<float>();

            results[j] = py::array_t<float>(size);
            result_ptrs[j] = results[j].mutable_data();
        }

        {
            py::gil_scoped_release release;  // Release GIL for all computations

            for (size_t j = 0; j < num_calls; j++) {
                float x_off = param_vec[j].y;
                float y_off = param_vec[j].z;
                float z_off = param_vec[j].w;
                float w_off = param_vec[j].v;
                float* r = result_ptrs[j];

                for (size_t i = 0; i < size; i++) {
                    float t_val = t(i);
                    float noise_val1 = noise1.GetNoise(x_off + t_val, y_off + t_val, z_off + t_val);
                    float noise_val2 = noise2.GetNoise(x_off + w_off, y_off + w_off, z_off + w_off);
                    float noise_val3 = noise1.GetNoise(x_off + t_val * 1.618f, y_off + t_val * 1.618f, z_off + w_off);
                    r[i] = 0.5f * noise_val1 + 0.3f * noise_val2 + 0.2f * noise_val3;
                }
            }
        }

        // Build Python list of results (needs GIL, which we've re-acquired)
        py::list result_list;
        for (size_t j = 0; j < num_calls; j++) {
            result_list.append(results[j]);
        }
        return result_list;
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
             py::arg("z_offset") = 0.0f, py::arg("w_offset") = 0.0f)
        .def("batch_generate", &Simplex5D::batch_generate,
             "Generate multiple noise arrays in one call (reduces Python-C++ overhead)",
             py::arg("t"), py::arg("params"));
}
