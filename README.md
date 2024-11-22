# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1 test output

```bash
=================================== test session starts ====================================
platform darwin -- Python 3.11.5, pytest-8.3.2, pluggy-1.5.0
rootdir: /Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 53 items / 2 deselected / 51 selected

tests/test_tensor_general.py ...................................................     [100%]

============================= 51 passed, 2 deselected in 6.07s =============================
```

## Task 3.2 test output

```bash
=================================== test session starts ====================================
platform darwin -- Python 3.11.5, pytest-8.3.2, pluggy-1.5.0
rootdir: /Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 53 items / 51 deselected / 2 selected

tests/test_tensor_general.py ..                                                      [100%]

============================= 2 passed, 51 deselected in 0.54s =============================
```

## Task 3.1 and 3.2 parallel analytics

```bash
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(174)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py (174)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        # raise NotImplementedError("Need to implement for Task 3.1")        |
        out_size = int(np.prod(out_shape))-----------------------------------| #2
                                                                             |
        if np.array_equal(out_strides, in_strides) and np.array_equal(       |
            out_shape, in_shape                                              |
        ):                                                                   |
            for i in prange(out_size):---------------------------------------| #3
                out[i] = fn(in_storage[i])                                   |
                                                                             |
        else:                                                                |
            for i in prange(out_size):---------------------------------------| #4
                out_index = np.zeros(len(out_shape), dtype=np.int32)---------| #0
                in_index = np.zeros(len(in_shape), dtype=np.int32)-----------| #1
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                in_pos = index_to_position(in_index, in_strides)             |
                out_pos = index_to_position(out_index, out_strides)          |
                out[out_pos] = fn(in_storage[in_pos])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #2, #3, #4, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--4 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #4) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#4).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(194) is hoisted out of the parallel loop labelled #4 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(195) is hoisted out of the parallel loop labelled #4 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(228)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py (228)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        # raise NotImplementedError("Need to implement for Task 3.1")      |
        out_size = int(np.prod(out_shape))---------------------------------| #9
                                                                           |
        can_use_direct_indexing = (                                        |
            np.array_equal(out_strides, a_strides)                         |
            and np.array_equal(a_strides, b_strides)                       |
            and np.array_equal(out_shape, a_shape)                         |
            and np.array_equal(a_shape, b_shape)                           |
        )                                                                  |
                                                                           |
        if can_use_direct_indexing:                                        |
            for i in prange(out_size):-------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                    |
                                                                           |
        else:                                                              |
            for i in prange(out_size):-------------------------------------| #10
                out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #5
                a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #6
                b_index = np.zeros(len(b_shape), dtype=np.int32)-----------| #7
                to_index(i, out_shape, out_index)                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                                                                           |
                out_pos = index_to_position(out_index, out_strides)        |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #9, #8, #10, #5, #6, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--5 (parallel)
   +--6 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--5 (serial)
   +--6 (serial)
   +--7 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 3 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(256) is hoisted out of the parallel loop labelled #10 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(257) is hoisted out of the parallel loop labelled #10 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(258) is hoisted out of the parallel loop labelled #10 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(292)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py (292)
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        reduce_dim: int,                                                 |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
        out_size = len(out)                                              |
        reduce_size = a_shape[reduce_dim]                                |
                                                                         |
        for i in prange(out_size):---------------------------------------| #11
            out_index = np.empty(len(out_shape), np.int32)               |
            to_index(i, out_shape, out_index)                            |
                                                                         |
            for s in range(reduce_size):                                 |
                out_index[reduce_dim] = s                                |
                j = index_to_position(out_index, a_strides)              |
                                                                         |
                out[i] = fn(out[i], a_storage[j])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(307) is hoisted out of the parallel loop labelled #11 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py
(319)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/jiayifan/mscb/mle/workspace/mod3-jiayifan-jf923/minitorch/fast_ops.py (319)
----------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                |
    out: Storage,                                                           |
    out_shape: Shape,                                                       |
    out_strides: Strides,                                                   |
    a_storage: Storage,                                                     |
    a_shape: Shape,                                                         |
    a_strides: Strides,                                                     |
    b_storage: Storage,                                                     |
    b_shape: Shape,                                                         |
    b_strides: Strides,                                                     |
) -> None:                                                                  |
    """NUMBA tensor matrix multiply function.                               |
                                                                            |
    Should work for any tensor shapes that broadcast as long as             |
                                                                            |
    ```                                                                     |
    assert a_shape[-1] == b_shape[-2]                                       |
    ```                                                                     |
                                                                            |
    Optimizations:                                                          |
                                                                            |
    * Outer loop in parallel                                                |
    * No index buffers or function calls                                    |
    * Inner loop should have no global writes, 1 multiply.                  |
                                                                            |
                                                                            |
    Args:                                                                   |
    ----                                                                    |
        out (Storage): storage for `out` tensor                             |
        out_shape (Shape): shape for `out` tensor                           |
        out_strides (Strides): strides for `out` tensor                     |
        a_storage (Storage): storage for `a` tensor                         |
        a_shape (Shape): shape for `a` tensor                               |
        a_strides (Strides): strides for `a` tensor                         |
        b_storage (Storage): storage for `b` tensor                         |
        b_shape (Shape): shape for `b` tensor                               |
        b_strides (Strides): strides for `b` tensor                         |
                                                                            |
    Returns:                                                                |
    -------                                                                 |
        None : Fills in `out`                                               |
                                                                            |
    """                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                  |
                                                                            |
    # TODO: Implement for Task 3.2.                                         |
    # raise NotImplementedError("Need to implement for Task 3.2")           |
                                                                            |
    batch_size = out_shape[0]                                               |
    m = out_shape[1]                                                        |
    n = out_shape[2]                                                        |
    k_size = a_shape[-1]  # Assuming a_shape[-1] == b_shape[-2]             |
                                                                            |
    total_output_elements = batch_size * m * n                              |
                                                                            |
    for idx in prange(total_output_elements):-------------------------------| #12
        batch = idx // (m * n)                                              |
        remaining = idx % (m * n)                                           |
        i = remaining // n                                                  |
        j = remaining % n                                                   |
                                                                            |
        sum = 0.0                                                           |
                                                                            |
        a_batch_offset = batch * a_batch_stride if a_shape[0] > 1 else 0    |
        a_i_offset = i * a_strides[1] if a_shape[1] > 1 else 0              |
                                                                            |
        b_batch_offset = batch * b_batch_stride if b_shape[0] > 1 else 0    |
        b_j_offset = j * b_strides[2] if b_shape[2] > 1 else 0              |
                                                                            |
        for k in range(k_size):                                             |
            a_k_offset = k * a_strides[2]                                   |
            a_pos = a_batch_offset + a_i_offset + a_k_offset                |
                                                                            |
            b_k_offset = k * b_strides[1]                                   |
            b_pos = b_batch_offset + b_k_offset + b_j_offset                |
                                                                            |
            sum += a_storage[a_pos] * b_storage[b_pos]                      |
                                                                            |
        out_batch_offset = batch * out_strides[0]                           |
        out_i_offset = i * out_strides[1]                                   |
        out_j_offset = j * out_strides[2]                                   |
        out_pos = out_batch_offset + out_i_offset + out_j_offset            |
                                                                            |
        out[out_pos] = sum                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #12).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Task 3.3 test output

```bash
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-jiayifan-jf923
configfile: pyproject.toml
plugins: env-1.1.4, hypothesis-6.54.0
collected 117 items / 60 deselected / 57 selected

tests/test_tensor_general.py .........................................................       [100%]

========================================= warnings summary =========================================
tests/test_tensor_general.py: 16 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 4274 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 13 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_args[cuda-fn1]
tests/test_tensor_general.py::test_two_args[cuda-fn1]
tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
tests/test_tensor_general.py::test_one_derivative[cuda-fn7]
tests/test_tensor_general.py::test_sum_practice2
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 9 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn7]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 27 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_sum_practice_other_dims
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================== 57 passed, 60 deselected, 4316 warnings in 280.43s (0:04:40) ===================
```

## Task 3.4 test output

```bash
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-jiayifan-jf923
configfile: pyproject.toml
plugins: env-1.1.4, hypothesis-6.54.0
collected 117 items / 110 deselected / 7 selected

tests/test_tensor_general.py .......                                                         [100%]

========================================= warnings summary =========================================
tests/test_tensor_general.py::test_mul_practice1
tests/test_tensor_general.py::test_mul_practice3
tests/test_tensor_general.py::test_mul_practice3
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 111 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice4
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice4
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice5
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 48 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 5 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 36 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 110 deselected, 140 warnings in 12.02s =========================
```

## Task 3.4 performance comparison

<img src="https://github.com/user-attachments/assets/ac2a5f72-6bc2-484a-a61f-095c7e4520f9" width="75%">


## Task 3.5

### Split - CPU

```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

```bash
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
Epoch  0  loss  6.142216359171325 correct 27    time 4.914520978927612
Epoch  10  loss  7.151082220193006 correct 40    time 0.1794719696044922
Epoch  20  loss  4.825603392249043 correct 40    time 0.17364811897277832
Epoch  30  loss  2.995454868304157 correct 38    time 0.1943058967590332
Epoch  40  loss  4.229215978494333 correct 43    time 0.166701078414917
Epoch  50  loss  2.5670879905997093 correct 50    time 0.14772391319274902
Epoch  60  loss  2.359048919805417 correct 49    time 0.14631891250610352
Epoch  70  loss  3.152226887804708 correct 49    time 0.16809701919555664
Epoch  80  loss  2.207566796123072 correct 49    time 0.15744996070861816
Epoch  90  loss  1.8804356532841426 correct 46    time 0.17449116706848145
Epoch  100  loss  1.465442224584624 correct 48    time 0.17807888984680176
Epoch  110  loss  0.7255380821359061 correct 50    time 0.16880512237548828
Epoch  120  loss  1.514123554487811 correct 50    time 0.15972900390625
Epoch  130  loss  1.0279521773075158 correct 50    time 0.16427922248840332
Epoch  140  loss  0.8824397817789804 correct 50    time 0.1681349277496338
Epoch  150  loss  1.2072266999013406 correct 49    time 0.17404699325561523
Epoch  160  loss  1.2165690182071467 correct 48    time 0.17168593406677246
Epoch  170  loss  1.2404744503154785 correct 49    time 0.14839816093444824
Epoch  180  loss  0.4375467999782218 correct 49    time 0.15937399864196777
Epoch  190  loss  2.249424629865193 correct 49    time 0.15543889999389648
Epoch  200  loss  0.9199521295348021 correct 50    time 0.16674304008483887
Epoch  210  loss  1.1932391249982246 correct 49    time 0.15001988410949707
Epoch  220  loss  1.9921936014082489 correct 49    time 0.15644001960754395
Epoch  230  loss  0.8712316854665967 correct 49    time 0.16956806182861328
Epoch  240  loss  1.025835169027129 correct 49    time 0.18836712837219238
Epoch  250  loss  0.6448627891660281 correct 50    time 0.16182780265808105
Epoch  260  loss  0.8795240302552523 correct 49    time 0.15660619735717773
Epoch  270  loss  1.2941402340225447 correct 49    time 0.15825104713439941
Epoch  280  loss  0.9443086911461762 correct 49    time 0.1542809009552002
Epoch  290  loss  0.07285331057762941 correct 49    time 0.1950697898864746
Epoch  300  loss  0.7665508165807611 correct 48    time 0.17876815795898438
Epoch  310  loss  0.8382232320656604 correct 49    time 0.1519758701324463
Epoch  320  loss  0.8378669868100713 correct 50    time 0.155364990234375
Epoch  330  loss  0.2787244383260293 correct 50    time 0.1526322364807129
Epoch  340  loss  0.33000035279594997 correct 49    time 0.15353608131408691
Epoch  350  loss  0.9914727952702612 correct 49    time 0.1821889877319336
Epoch  360  loss  0.5722465360447443 correct 50    time 0.15696167945861816
Epoch  370  loss  0.820890522831572 correct 49    time 0.15984487533569336
Epoch  380  loss  0.9508871577230548 correct 49    time 0.15281271934509277
Epoch  390  loss  1.690221117227896 correct 49    time 0.1539618968963623
Epoch  400  loss  0.302089997682567 correct 49    time 0.15572309494018555
Epoch  410  loss  0.511068104616592 correct 49    time 0.154710054397583
Epoch  420  loss  1.1264032976405023 correct 49    time 0.1589980125427246
Epoch  430  loss  0.040463263276197384 correct 49    time 0.16756200790405273
Epoch  440  loss  0.30985223566699416 correct 50    time 0.16249489784240723
Epoch  450  loss  0.4121136911660879 correct 49    time 0.17666888236999512
Epoch  460  loss  0.9115397847588131 correct 49    time 0.1672987937927246
Epoch  470  loss  1.6025126629071185 correct 47    time 0.1530900001525879
Epoch  480  loss  0.8701292494154055 correct 49    time 0.15537571907043457
Epoch  490  loss  0.1898661494391321 correct 50    time 0.17412185668945312
===== Average time per epoch: 0.17470s
```

### Split - GPU

```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

```bash
Epoch  0  loss  5.208522327160182 correct 33    time 5.406404972076416
Epoch  10  loss  3.507907648795703 correct 33    time 2.149547815322876
Epoch  20  loss  4.896481460150728 correct 46    time 2.0698349475860596
Epoch  30  loss  4.503574554146671 correct 48    time 1.9908390045166016
Epoch  40  loss  3.0270134255929237 correct 40    time 2.534402370452881
Epoch  50  loss  5.526626887492435 correct 49    time 2.0362589359283447
Epoch  60  loss  1.8906418208750095 correct 48    time 1.9990062713623047
Epoch  70  loss  3.3833505250864118 correct 48    time 2.013277530670166
Epoch  80  loss  1.494539033227845 correct 49    time 2.6935043334960938
Epoch  90  loss  1.0287490621247142 correct 47    time 1.9966702461242676
Epoch  100  loss  0.661524680667529 correct 49    time 1.9987983703613281
Epoch  110  loss  0.8329333060597504 correct 50    time 2.0423648357391357
Epoch  120  loss  1.4699962827036501 correct 50    time 2.4726812839508057
Epoch  130  loss  1.4513372035411263 correct 49    time 1.9796597957611084
Epoch  140  loss  1.024149402414733 correct 50    time 2.06718111038208
Epoch  150  loss  1.373477349318704 correct 49    time 2.32342267036438
Epoch  160  loss  1.2253353748696976 correct 49    time 2.0526058673858643
Epoch  170  loss  1.4472888363652536 correct 50    time 2.023062229156494
Epoch  180  loss  0.6768886642259382 correct 49    time 2.0090372562408447
Epoch  190  loss  0.6298890176781058 correct 50    time 2.6433217525482178
Epoch  200  loss  0.7491039825488871 correct 50    time 2.0301153659820557
Epoch  210  loss  0.32305125407622903 correct 50    time 2.029545783996582
Epoch  220  loss  0.46951823491018313 correct 50    time 2.0799365043640137
Epoch  230  loss  1.155810677178211 correct 50    time 2.214097738265991
Epoch  240  loss  0.4496020042398263 correct 50    time 2.2942590713500977
Epoch  250  loss  0.45654511309003243 correct 50    time 2.0557401180267334
Epoch  260  loss  0.4554472938062173 correct 50    time 1.9831664562225342
Epoch  270  loss  0.14496746114193992 correct 50    time 2.481708288192749
Epoch  280  loss  0.31166665247232506 correct 50    time 2.051166296005249
Epoch  290  loss  1.105738831222697 correct 50    time 1.995535135269165
Epoch  300  loss  0.11324553014379345 correct 50    time 2.0110509395599365
Epoch  310  loss  0.38421425317310975 correct 50    time 2.9105663299560547
Epoch  320  loss  0.23205092820493034 correct 50    time 2.016066551208496
Epoch  330  loss  0.5811953370534425 correct 50    time 2.4196200370788574
Epoch  340  loss  0.20435430680154734 correct 50    time 3.1773664951324463
Epoch  350  loss  0.2468893723123542 correct 50    time 2.2142741680145264
Epoch  360  loss  0.40259527296247855 correct 50    time 2.7199363708496094
Epoch  370  loss  0.35354578132712433 correct 50    time 2.00788950920105
Epoch  380  loss  0.4482808180634089 correct 50    time 2.0041494369506836
Epoch  390  loss  0.5828088217582617 correct 50    time 2.0792741775512695
Epoch  400  loss  0.06008244482097805 correct 50    time 2.8624374866485596
Epoch  410  loss  0.3337720139356013 correct 50    time 1.969252586364746
Epoch  420  loss  0.47469871173587846 correct 50    time 2.080368757247925
Epoch  430  loss  0.37140299690440026 correct 50    time 2.07766056060791
Epoch  440  loss  0.5493489382033938 correct 50    time 2.8696329593658447
Epoch  450  loss  0.20497920310027062 correct 50    time 2.0666749477386475
Epoch  460  loss  0.2731248199951767 correct 50    time 1.994192361831665
Epoch  470  loss  0.34234530809005914 correct 50    time 2.0365495681762695
Epoch  480  loss  0.3134272268371898 correct 50    time 2.901521921157837
Epoch  490  loss  0.012443053053214135 correct 50    time 1.9994630813598633
===== Average time per epoch: 2.19603s
```

### Bigger Split - CPU

```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05
```

```bash
Epoch  0  loss  6.422991802481079 correct 26    time 5.43889307975769
Epoch  10  loss  3.2079102175044216 correct 45    time 0.543367862701416
Epoch  20  loss  2.7992329780385257 correct 45    time 0.5068438053131104
Epoch  30  loss  2.553164726508965 correct 43    time 0.49578022956848145
Epoch  40  loss  1.6877860678267647 correct 49    time 0.49694108963012695
Epoch  50  loss  2.2573877272859266 correct 49    time 0.4969460964202881
Epoch  60  loss  2.5457615641355638 correct 46    time 0.5079898834228516
Epoch  70  loss  2.125448548599129 correct 45    time 0.49399304389953613
Epoch  80  loss  1.659230489522647 correct 48    time 0.5114061832427979
Epoch  90  loss  0.6037472829936517 correct 49    time 0.5012948513031006
Epoch  100  loss  3.284821626730057 correct 48    time 0.5096402168273926
Epoch  110  loss  0.4282906326423258 correct 45    time 0.5214650630950928
Epoch  120  loss  1.0176399620256131 correct 49    time 0.5140440464019775
Epoch  130  loss  1.1737906243109448 correct 50    time 0.4981210231781006
Epoch  140  loss  1.0375335128998329 correct 49    time 0.5161440372467041
Epoch  150  loss  1.4688641730849934 correct 49    time 0.49690985679626465
Epoch  160  loss  1.2717366338690406 correct 49    time 0.4964258670806885
Epoch  170  loss  0.9791065607538051 correct 50    time 0.48422884941101074
Epoch  180  loss  0.19829191236362842 correct 50    time 0.5195538997650146
Epoch  190  loss  0.7075085769827352 correct 50    time 0.49278903007507324
Epoch  200  loss  0.6971107614106494 correct 50    time 0.5161709785461426
Epoch  210  loss  0.3739471881717486 correct 50    time 0.49801087379455566
Epoch  220  loss  0.08639357452023164 correct 50    time 0.49401283264160156
Epoch  230  loss  0.8313927299068601 correct 50    time 0.5001039505004883
Epoch  240  loss  0.2003953191369365 correct 47    time 0.4970581531524658
Epoch  250  loss  0.8025834300819539 correct 50    time 0.5023849010467529
Epoch  260  loss  1.00992808114972 correct 50    time 0.49478697776794434
Epoch  270  loss  0.6403865985400912 correct 50    time 0.49037909507751465
Epoch  280  loss  0.17102598387080037 correct 50    time 0.5132579803466797
Epoch  290  loss  0.3049012697656507 correct 49    time 0.48801708221435547
Epoch  300  loss  0.8480573270454101 correct 49    time 0.510718822479248
Epoch  310  loss  0.7217795221087089 correct 50    time 0.5112912654876709
Epoch  320  loss  0.8586493889487432 correct 50    time 0.48238492012023926
Epoch  330  loss  0.4425965502115372 correct 50    time 0.509221076965332
Epoch  340  loss  0.13362154965289327 correct 50    time 0.4886610507965088
Epoch  350  loss  0.08022081406701458 correct 50    time 0.4897778034210205
Epoch  360  loss  0.38828984035238157 correct 50    time 0.5067181587219238
Epoch  370  loss  0.24736892906953864 correct 49    time 0.5006279945373535
Epoch  380  loss  0.8285609802951657 correct 50    time 0.5055170059204102
Epoch  390  loss  0.5113850518544496 correct 50    time 0.49128293991088867
Epoch  400  loss  0.035553065793423905 correct 50    time 0.5069749355316162
Epoch  410  loss  0.2488224789717854 correct 50    time 0.5124759674072266
Epoch  420  loss  0.22225156150526412 correct 50    time 0.49591588973999023
Epoch  430  loss  0.2637726341234561 correct 50    time 0.48954010009765625
Epoch  440  loss  0.4795364224275531 correct 50    time 0.49800610542297363
Epoch  450  loss  0.16999469106631757 correct 50    time 0.49534177780151367
Epoch  460  loss  0.22203294202579127 correct 50    time 0.5022947788238525
Epoch  470  loss  0.01913058248889387 correct 50    time 0.4907357692718506
Epoch  480  loss  0.6330043483017931 correct 50    time 0.5048348903656006
Epoch  490  loss  0.11935372125156841 correct 50    time 0.4995579719543457
===== Average time per epoch: 0.51011s
```

### Bigger Split - GPU

```bash
Epoch  0  loss  12.31899604699696 correct 39    time 5.817668199539185
Epoch  10  loss  4.158542582449523 correct 44    time 2.7606680393218994
Epoch  20  loss  6.866373327150705 correct 20    time 2.8047144412994385
Epoch  30  loss  2.7199657074273653 correct 48    time 3.2963337898254395
Epoch  40  loss  1.416484881198786 correct 50    time 2.7371468544006348
Epoch  50  loss  1.431638537639694 correct 48    time 3.0519773960113525
Epoch  60  loss  1.3995891239103226 correct 48    time 2.7697765827178955
Epoch  70  loss  1.54350698674784 correct 47    time 2.734764814376831
Epoch  80  loss  0.3317580644168439 correct 48    time 3.6856799125671387
Epoch  90  loss  0.5494728599542456 correct 50    time 2.758676528930664
Epoch  100  loss  0.6656223199212313 correct 48    time 2.8084771633148193
Epoch  110  loss  0.39631261145350494 correct 50    time 2.917909860610962
Epoch  120  loss  1.382049259792356 correct 48    time 2.720076322555542
Epoch  130  loss  1.4558916857555495 correct 48    time 3.5105443000793457
Epoch  140  loss  0.678950814255002 correct 48    time 2.8155617713928223
Epoch  150  loss  3.041267965567782 correct 47    time 2.7776875495910645
Epoch  160  loss  0.920629134733652 correct 48    time 3.558584213256836
Epoch  170  loss  0.42515299067886236 correct 49    time 2.7966794967651367
Epoch  180  loss  0.1246538422581099 correct 50    time 3.2846922874450684
Epoch  190  loss  1.0864441419595616 correct 50    time 2.7068636417388916
Epoch  200  loss  1.1871070567024904 correct 48    time 2.690479278564453
Epoch  210  loss  0.3882922884294505 correct 48    time 3.5868284702301025
Epoch  220  loss  0.8276215623950641 correct 48    time 2.778022050857544
Epoch  230  loss  0.6312217996460742 correct 48    time 2.780054807662964
Epoch  240  loss  0.34320825213654915 correct 50    time 3.4702694416046143
Epoch  250  loss  1.945356010709956 correct 48    time 2.8071515560150146
Epoch  260  loss  1.5885355791349498 correct 49    time 2.94094181060791
Epoch  270  loss  1.614579311233957 correct 49    time 2.722029447555542
Epoch  280  loss  0.35277046253458777 correct 50    time 2.764355182647705
Epoch  290  loss  0.7698873442046736 correct 49    time 3.05537486076355
Epoch  300  loss  1.0714491537313555 correct 50    time 2.7245399951934814
===== Average time per epoch: 3.05273s
```


### XOR - CPU

```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

```bash
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
Epoch  0  loss  7.705237825945431 correct 27    time 5.010254144668579
Epoch  10  loss  5.279981850068566 correct 38    time 0.2045729160308838
Epoch  20  loss  5.116614275229045 correct 42    time 0.20190906524658203
Epoch  30  loss  6.468888363743627 correct 42    time 0.14989995956420898
Epoch  40  loss  4.117155408312074 correct 46    time 0.1599869728088379
Epoch  50  loss  1.950473449889314 correct 46    time 0.17133212089538574
Epoch  60  loss  4.137401872117712 correct 44    time 0.17698311805725098
Epoch  70  loss  3.3930379219284217 correct 44    time 0.17471623420715332
Epoch  80  loss  3.3965488277147173 correct 44    time 0.14609622955322266
Epoch  90  loss  5.66349498466646 correct 42    time 0.15252470970153809
Epoch  100  loss  1.8508526733377049 correct 48    time 0.15781497955322266
Epoch  110  loss  3.115131577933006 correct 49    time 0.1701059341430664
Epoch  120  loss  3.022644938326622 correct 48    time 0.15878915786743164
Epoch  130  loss  1.8233799377373392 correct 49    time 0.20043492317199707
Epoch  140  loss  3.1017104686761305 correct 49    time 0.16130995750427246
Epoch  150  loss  1.0684735058131698 correct 48    time 0.17567801475524902
Epoch  160  loss  2.283140672612467 correct 49    time 0.15607810020446777
Epoch  170  loss  1.3635302330445862 correct 49    time 0.15189290046691895
Epoch  180  loss  1.001334423189148 correct 50    time 0.17522478103637695
Epoch  190  loss  2.762919060931242 correct 46    time 0.17316102981567383
Epoch  200  loss  1.6434763292729146 correct 50    time 0.17549586296081543
Epoch  210  loss  1.0256699664146938 correct 48    time 0.16391611099243164
Epoch  220  loss  2.3701188803877686 correct 46    time 0.1572270393371582
Epoch  230  loss  0.8089931335586341 correct 48    time 0.17283892631530762
Epoch  240  loss  0.4333309826990776 correct 49    time 0.15735387802124023
Epoch  250  loss  1.9449304481268193 correct 50    time 0.16112589836120605
Epoch  260  loss  2.2876251152113776 correct 47    time 0.1631031036376953
Epoch  270  loss  1.5755074847189214 correct 49    time 0.18126916885375977
Epoch  280  loss  0.9181725834392721 correct 48    time 0.16225099563598633
Epoch  290  loss  0.9117540178373682 correct 48    time 0.16055512428283691
Epoch  300  loss  1.990738626254072 correct 48    time 0.16529607772827148
Epoch  310  loss  2.6068139730637383 correct 49    time 0.15590310096740723
Epoch  320  loss  0.3453377324902136 correct 50    time 0.1744687557220459
Epoch  330  loss  0.3677434800302188 correct 50    time 0.14988064765930176
Epoch  340  loss  0.525379470451905 correct 50    time 0.17518401145935059
Epoch  350  loss  1.0180108427508243 correct 50    time 0.15996003150939941
Epoch  360  loss  0.7070789700161885 correct 47    time 0.15384411811828613
Epoch  370  loss  0.5654725466186837 correct 50    time 0.15242290496826172
Epoch  380  loss  1.2257291678961761 correct 50    time 0.15289926528930664
Epoch  390  loss  0.6678506624529865 correct 50    time 0.1545097827911377
Epoch  400  loss  0.9552810915463156 correct 48    time 0.15674281120300293
Epoch  410  loss  1.2987569068487288 correct 50    time 0.1527869701385498
Epoch  420  loss  0.5249620187298347 correct 48    time 0.1520519256591797
Epoch  430  loss  1.9302235045283294 correct 49    time 0.15434598922729492
Epoch  440  loss  1.5221593866213061 correct 48    time 0.16007566452026367
Epoch  450  loss  0.5236507907258559 correct 48    time 0.1460130214691162
Epoch  460  loss  2.0460392800663048 correct 48    time 0.1512281894683838
Epoch  470  loss  0.9588744410392067 correct 50    time 0.15442395210266113
Epoch  480  loss  1.1642570512322619 correct 50    time 0.15449070930480957
Epoch  490  loss  0.6442656252454578 correct 49    time 0.1467578411102295
===== Average time per epoch: 0.17435s
```

### XOR - GPU

```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

```bash
Epoch  0  loss  7.2335876544772 correct 36    time 4.370588302612305
Epoch  10  loss  5.146805378758415 correct 34    time 1.9490625858306885
Epoch  20  loss  4.3740364515789585 correct 49    time 1.9564931392669678
Epoch  30  loss  2.7842584640530195 correct 50    time 1.9127659797668457
Epoch  40  loss  2.263084934765328 correct 49    time 1.8908333778381348
Epoch  50  loss  2.240449325834092 correct 49    time 1.9985668659210205
Epoch  60  loss  1.4819851444673726 correct 47    time 1.907942771911621
Epoch  70  loss  1.8438537105069612 correct 49    time 1.889615535736084
Epoch  80  loss  1.28151722108712 correct 50    time 1.9561247825622559
Epoch  90  loss  1.0835437076534338 correct 50    time 1.9091176986694336
Epoch  100  loss  1.149454636758796 correct 50    time 1.8911569118499756
Epoch  110  loss  0.9730675788771 correct 50    time 1.9610893726348877
Epoch  120  loss  0.813501969786811 correct 49    time 1.9030568599700928
Epoch  130  loss  0.48987780074215515 correct 50    time 1.8933358192443848
Epoch  140  loss  0.9702639030515064 correct 50    time 1.9655306339263916
Epoch  150  loss  0.3689305611792789 correct 50    time 1.8974413871765137
Epoch  160  loss  1.2524676785562654 correct 50    time 1.8963623046875
Epoch  170  loss  0.6712116746655054 correct 50    time 1.9558799266815186
Epoch  180  loss  0.2921604764962731 correct 49    time 1.8946754932403564
Epoch  190  loss  0.8529615013209411 correct 50    time 1.8893980979919434
Epoch  200  loss  0.38634599360847377 correct 50    time 1.888514757156372
===== Average time per epoch: 2.03703s
```

### Simple - CPU

```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --RATE 0.05
```

```bash
Epoch  0  loss  1.2031371451411932 correct 45    time 5.475275039672852
Epoch  10  loss  0.4650013978207789 correct 50    time 0.5053930282592773
Epoch  20  loss  0.8180409538475393 correct 48    time 0.4941880702972412
Epoch  30  loss  0.05559680794675104 correct 49    time 0.5069589614868164
Epoch  40  loss  0.5166225550753641 correct 49    time 0.4855828285217285
Epoch  50  loss  0.3267227599471889 correct 50    time 0.5066559314727783
Epoch  60  loss  0.05008406504354674 correct 48    time 0.4984598159790039
Epoch  70  loss  0.8022472090471822 correct 50    time 0.47843098640441895
Epoch  80  loss  0.04013598679996832 correct 50    time 0.5048720836639404
Epoch  90  loss  0.250031245081278 correct 49    time 0.4874119758605957
Epoch  100  loss  0.2539968993463861 correct 49    time 0.4889681339263916
Epoch  110  loss  0.06392254324656534 correct 49    time 0.5273170471191406
Epoch  120  loss  0.0015848481238703708 correct 49    time 0.659998893737793
Epoch  130  loss  0.7002220473612008 correct 50    time 0.4878690242767334
Epoch  140  loss  0.07414380381022057 correct 50    time 0.5039830207824707
Epoch  150  loss  0.7070739404823959 correct 50    time 0.5062782764434814
Epoch  160  loss  0.6209864553947239 correct 50    time 0.5019910335540771
Epoch  170  loss  0.5314055729055233 correct 49    time 0.4941532611846924
Epoch  180  loss  0.02758338114846396 correct 50    time 0.5506460666656494
Epoch  190  loss  0.16968227334176447 correct 50    time 0.506633996963501
Epoch  200  loss  0.4438293819579518 correct 50    time 0.49611878395080566
Epoch  210  loss  0.2525144692518685 correct 50    time 0.5677738189697266
Epoch  220  loss  0.5837348526374141 correct 50    time 0.5050270557403564
Epoch  230  loss  0.5446850813314053 correct 50    time 0.511481761932373
Epoch  240  loss  0.04818097981304568 correct 50    time 0.49933791160583496
Epoch  250  loss  0.5839929958275744 correct 50    time 0.5082459449768066
Epoch  260  loss  0.004703091108867517 correct 50    time 0.4989910125732422
Epoch  270  loss  0.00015246912153545056 correct 50    time 0.5081710815429688
Epoch  280  loss  0.6408246446156973 correct 50    time 0.4948310852050781
Epoch  290  loss  0.0013149049091517398 correct 50    time 0.4948999881744385
Epoch  300  loss  0.00036714373381952864 correct 50    time 0.5004019737243652
Epoch  310  loss  0.2722325416571056 correct 50    time 0.5342111587524414
Epoch  320  loss  0.011400274698610137 correct 50    time 0.48828721046447754
Epoch  330  loss  0.16230125007192875 correct 50    time 0.49410009384155273
Epoch  340  loss  0.26965873953910957 correct 50    time 0.4893622398376465
Epoch  350  loss  0.5026159647093906 correct 50    time 0.5100858211517334
Epoch  360  loss  0.39396609177587216 correct 50    time 0.4960508346557617
Epoch  370  loss  0.003576684401312803 correct 50    time 0.4846508502960205
Epoch  380  loss  5.0512629348077714e-05 correct 50    time 0.49189281463623047
Epoch  390  loss  0.00027545638973749937 correct 50    time 0.49846625328063965
Epoch  400  loss  2.7751039369870922e-05 correct 50    time 0.4929077625274658
Epoch  410  loss  0.0025841331534473123 correct 50    time 0.5065970420837402
Epoch  420  loss  0.2624023673199203 correct 50    time 0.5026760101318359
Epoch  430  loss  0.2587636635290611 correct 50    time 0.5676929950714111
Epoch  440  loss  0.23924639751675772 correct 50    time 0.4954648017883301
Epoch  450  loss  0.3972199832221568 correct 50    time 0.5059430599212646
Epoch  460  loss  0.06990897924943934 correct 50    time 0.493135929107666
Epoch  470  loss  0.03649899036957525 correct 50    time 0.49584102630615234
Epoch  480  loss  0.0007323669947173033 correct 50    time 0.5123560428619385
Epoch  490  loss  0.4507068720833451 correct 50    time 0.48406291007995605
===== Average time per epoch: 0.51377s
```

### Simple - GPU

```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --RATE 0.05
```

```bash
Epoch  0  loss  5.260727037883097 correct 46    time 5.9863481521606445
Epoch  10  loss  0.853969140821446 correct 47    time 2.138427972793579
Epoch  20  loss  1.576332736409127 correct 50    time 2.0229456424713135
Epoch  30  loss  1.0637631453095688 correct 49    time 1.980865478515625
Epoch  40  loss  1.3212846899980377 correct 50    time 2.807464599609375
Epoch  50  loss  0.1132686202550572 correct 49    time 2.025202989578247
Epoch  60  loss  0.7608573836889283 correct 49    time 1.97646164894104
Epoch  70  loss  1.1682291030602545 correct 49    time 2.145169496536255
Epoch  80  loss  0.1224772850436645 correct 49    time 2.199568033218384
Epoch  90  loss  0.11233267991992361 correct 49    time 1.9832308292388916
Epoch  100  loss  1.4268202906384375 correct 50    time 1.9786429405212402
Epoch  110  loss  1.200710044753487 correct 50    time 2.833139419555664
Epoch  120  loss  0.5101635513059662 correct 49    time 1.969498634338379
Epoch  130  loss  0.054016555888374174 correct 49    time 2.000128746032715
Epoch  140  loss  1.1466084250971131 correct 50    time 2.0413286685943604
Epoch  150  loss  0.399352756123355 correct 49    time 2.740445137023926
Epoch  160  loss  0.1489130088971785 correct 49    time 1.9778478145599365
Epoch  170  loss  0.06580348520915277 correct 49    time 2.038814067840576
Epoch  180  loss  1.0990729554303393 correct 50    time 2.172300100326538
Epoch  190  loss  0.018761219841319017 correct 50    time 1.9879462718963623
Epoch  200  loss  1.3500443466652405 correct 49    time 1.9923725128173828
===== Average time per epoch: 2.33325s
```