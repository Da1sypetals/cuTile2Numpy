/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"
#include <cuda.h>

struct DriverApi;

Status cuda_helper_init(PyObject* m);

const char* get_cuda_error(const DriverApi*, CUresult res);

void try_cuInit(const DriverApi*);

Status check_driver_version(const DriverApi*, int minimum_version);
