# learn-cuda

Terms related to GPUs:

1. Threads,blocks,grid
2. GEMM = General Matrix Multiplication
3. SGEMM = Single precision General Matrix multiplication
4. cpi/host/functions vs gpu/device/kernels
5. CPU is host that executes functions
6. GPU is a device that execeutes kernels ( GPU Functions )


## Execution 

Host => CPU => Uses RAM
Device => GPU => Uses chip VRAM

1. copy input from host to device
2. load GPU program & execute using the transferred on-device data
3. copy results from device back to host

## Naming schemes

`h_A` - HOST for var `A`
`d_A` - HOST for var `A`
`__global__` - cuda kernels ( Host can call these ) , runs on device
`__device__` - small job only called by GPU , runs on device
`__host__` - called from host , runs on host 

## Mem Management

`cudaMalloc` - mem alloc on VRAM only ( Global Mem )
`cudaMemcpy` - copy mem from device to host , host to device or device to device. 
`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, or `cudaMemcpyDeviceToDevice`
`cudaFree` - frees memory on the device
`cudaDeviceSynchronize`- Wait CPU untill all GPU operations are complete

## Memories in GPU

1. Global Memory 
    - main mem of GPU , accessible to all threads 
    - Slowest but largest
    - eg. arrays & large datasets
2. Shared Memory
    - mem shared by all  threads in a block
    - very fast & small
    - high frequency shared data
    - eg.temp vars
3. Registers
    - fastest , private to a thread
    - local vars
    - eg. loop counters
4. Constant memory 
    - read-only mem
    - cached 
    - used for data not changing during kernel execution
    - eg. constants 
5. Local memory
    - spilling use case ( registers arent enough )
    - slow ( stored in global mem )
    - eg. large arrays spilling out of registers

## `nvcc` compiler

- Host code 
    - modified to run kernels
    - compiled to x86 binary
- Device code
    - compiled to PTX (`parallel thread execution`)
    - stable across multiple GPU generations
- JIT 
    - PTX into native GPU ist
    - Allow forward compt.

## Hirerachy 

1. kernel executes in a thread
2. threadds grouped into blocks 
3. blocks to grid
4. kernel executes as grid of blocks of threads

`gridDim` - no. of blocks in grid
`blockIdx` - idx of block in grid 
`blockDim` - no. of threads in a block
`threadIdx` - idx of the thread in block

## Threads 

- thread has a set of reg
- each thread would do a single op
- ### Thread ID
    - This ID allows you to access specific data or perform different operations based on the thread's position within the block.
    - `threadIdx` ( 3-component vector  (x,y,z) pos of thread in block)
    - eg. 1D block of 256 threads - threadIdx.x ranges `0..255`
    - indexing is neccesary given that a thread should be assigned a task and multiple threads must not do the same task

## Warps

- Wrap is inside block ( 32 threads parallelization )
- 4 Wrap schedulers per Streaming multiprocessor 
- SM - main compute unit inside the nvidia GPU 
    1. CUDA cores
    2. Warp schedulers 
    3. Registers 
    4. special function units
    5. load/store units

## Blocks

- each block has shared memory ( shared with all threads in the block)
- execute smae code with diff. data
- `blockDim` ( 3 comps x,y,z ) specifies dimension of the box
- `blockIdx` ( 3 comps x,y,z ) block's position within the grid

## Grids

- during kernel execution , threads in blocks can access global memory
- a bunch of blocks.
- `gridDim` (x,y,z) 
- `gridDim.x` = 10 if 10 blocks in x-direction

unique global thread id ( access elements in 1D array )
```cpp
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

# Helper types

- `dim3`
    - simple way to specify 3D dimensions 
    - used for grid & block sizes
    - eg.
    - ```cpp
        dim3 blockSize(16, 16, 1);  // 16x16x1 threads in block
        dim3 gridSize(8, 8, 1);     // 8x8x1 blocks in grid
      ```

- `<<<>>>`
    - not CPP templates new syntax introduced by CUDA
    - used to configure & launch kernels on GPU
    - 1st arg - no. of blocks in grid 
    - 2nd arg - no. of threads per block
