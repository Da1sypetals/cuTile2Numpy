// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_helper.h"
#include "cuda_loader.h"


const char* get_cuda_error(const DriverApi* driver, CUresult res) {
    const char* str = nullptr;
    driver->cuGetErrorString(res, &str);
    return str ? str : "Unknown error";
}

Status check_driver_version(const DriverApi* driver, int minimum_version) {
    int version;
    CUresult res = driver->cuDriverGetVersion(&version);
    if (res != CUDA_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "cuDriverGetVersion: %s", get_cuda_error(driver, res));
        return ErrorRaised;
    }
    if (version < minimum_version) {
        int major = version / 1000;
        int minor = (version % 1000) / 10;
        int required_major = minimum_version / 1000;
        PyErr_Format(PyExc_RuntimeError,
                     "Minimum driver version required is %d.0, got %d.%d",
                     required_major, major, minor);
        return ErrorRaised;
    }
    return OK;
}

PyObject* get_max_grid_size(PyObject *self, PyObject *args) {
    int device_id;
    if (!PyArg_ParseTuple(args, "i", &device_id))
        return NULL;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return NULL;

    CUdevice dev;
    CUresult res = (*driver)->cuDeviceGet(&dev, device_id);
    if (res != CUDA_SUCCESS)
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(*driver, res));

    int max_grid_size[3];
    for (int i = 0; i < 3; ++i) {
        res = (*driver)->cuDeviceGetAttribute(&max_grid_size[i],
            static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i),
            dev);
        if (res != CUDA_SUCCESS) {
            return PyErr_Format(PyExc_RuntimeError,
                                "cuDeviceGetAttribute: %s", get_cuda_error(*driver, res));
        }
    }
    return Py_BuildValue("(iii)", max_grid_size[0], max_grid_size[1], max_grid_size[2]);
}

PyObject* get_compute_capability(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int major, minor;
    CUdevice dev;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(d, res));
    }
    res = d->cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGetAttribute: %s", get_cuda_error(d, res));
    }
    res = d->cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGetAttribute: %s", get_cuda_error(d, res));
    }
    return Py_BuildValue("(ii)", major, minor);
}

PyObject* get_driver_version(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int major, minor;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuDriverGetVersion(&major);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDriverGetVersion: %s", get_cuda_error(d, res));
    }
    minor = (major % 1000) / 10;
    major = major / 1000;
    return Py_BuildValue("(ii)", major, minor);
}

static PyMethodDef functions[] = {
    {"get_compute_capability", get_compute_capability, METH_NOARGS,
        "Get compute capability of the default CUDA device"},
    {"get_driver_version", get_driver_version, METH_NOARGS,
        "Get the cuda driver version"},
    {"_get_max_grid_size", get_max_grid_size, METH_VARARGS,
        "Get max grid size of a CUDA device, given device id"},
    NULL
};

Status cuda_helper_init(PyObject* m) {
    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    return OK;
}
