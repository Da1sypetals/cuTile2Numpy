<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Lift the ban on recursive helper function calls. Instead, add a limit on recursion depth.
- Add a new exception class `TileRecursionError`, thrown at compile time when the recursion limit
  is reached during function call inlining.
- Include a full cuTile traceback in error messages. Improve formatting of code locations:
  include function names, remove unnecessary characters to reduce line lengths.
- Expose the `TileError` base class in the public API.

