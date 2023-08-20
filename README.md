# About HEIR
HEIR is an end-to-end FHE compiler to compile high-level input 
C programs and emits to efficient FHE implementations. Currently, 
based on the arithmetic and Boolean nature of the C
instructions, HEIR can transform these instructions into 
homomorphic operators supported in either Ring-LWE based schemes (CKKS)
or LWE based schemes (TFHE).

> **Note:
> This repository only contains the IR part of HEIR, to evaluate the
> benchmarks, please use the [Docker](https://hub.docker.com/repository/docker/zhaozian/heir/general)
> version of HEIR.
# Repository's Structure
The repository is organized as follow:
```
cmake             – configuration files for the CMake build system
include           – header (.h) and TableGen (.td) files
 └ IR               – contains HEIR dialect definitions
 └ Passes           – contains the definitions of the different transformations
src               – source files (.cpp)
 └ IR               – implementations of additional dialect-specific functionality
 └ Passes           – implementations of the different transformations
 └ tools            – sources for the main commandline interface
format_assistant  – scripts for integrating Middle-End IR with Back-End FHE library
tools             – pre-built Front-End and Middle-End CLI
HEIR_plot         – scripts for results comparison and plots generation 
benchmarks        – orignal version for artifact evaluation
HEIR_full_bench   – full input programs for plot generation
```

# Using HEIR
### Front-End
HEIR uses Polygeist CLI `cgeist` as the Front-End to transform
the input C program into `*.mlir` file. Please use the 
following default parameters to execute the tool.
```sh
./tools/cgeist $fileName$.c \
  -function=$functionName$ -S \
  -raise-scf-to-affine \
  --memref-fullrank -O0
```
### Middle-End
In Middle-End, HEIR uses `heir-opt` CLI to transform the input
MLIR program into programs with homomorphic operators 
reprsented in `emitc` dialect. There are three parameters for 
`heir-opt`:

+ **--branch**: Add this parameter when `if` insturction is 
called in the input C program.
+ **--affine-loop-unroll="unroll-full unroll-num-reps=4"**: 
Add this parameter to unroll all the `for` loop in the 
input program.
+ **--arith-emitc/--logic-emitc**: If the input program only includes arithmetic operations, use **--arith-emitc** for batching optimizations. Otherwise, use **--logic-emitc**.  

Next, HEIR uses `emitc-translate` to transform the MLIR file
into a C++ file:
```sh
./tools/emitc-translate $fileName$.mlir --mlir-to-cpp
```
### Back-End
The integration between Middle-End and Back-End is not yet 
well-implemented. If you require an executable, please use 
`format_assistant/halo_transmitter.py` script and manual 
compilation & linking for now.

# A Guide For HEIR Installation
> **Note**
> In this docker version, we pre-built the Front-End and Middle-End
> executables in `./tools`, i.e. `cgeist`, `heir-opt` and `emitc-translate`.  
## Build Polygeist Front-End
Start with ``HEIR`` directory.

Clone Polygeist from Github.
```sh
cd ..
git clone -b dev --recursive https://github.com/heir-compiler/Polygeist
cd Polygeist
```
Using unified LLVM, MLIR, Clang, and Polygeist build.
```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
```

## Build HEIR Middle-End
Start with ``HEIR`` directory.

Clone llvm-15 from Github. Note that LLVM-15 used for HEIR Middle-End is not compatiable with LLVM for building Polygeist.
```sh
cd ..
git clone -b release/15.x https://github.com/llvm/llvm-project.git
cd llvm-project
```
Build LLVM/MLIR.
```sh
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON
ninja -j N
```

Build HEIR.
```sh
cd ../../HEIR
mkdir build && cd build
cmake .. -DMLIR_DIR=/home/NDSS_Artifact/llvm-project/build/lib/cmake/mlir
cmake --build . --target all
```

## BUILD HALO Back-End
Start with ``HEIR`` directory.

We first build and install Microsoft SEAl to ``/home/NDSS_Artifact/mylibs/SEAL3.7``.
```sh
cd ..
git clone -b 3.7.1 https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/home/NDSS_Artifact/mylibs/SEAL3.7 -DSEAL_USE_INTEL_HEXL=ON
cmake --build build
sudo cmake --install build
```
Build HALO with Microsoft SEAL.
```sh
cd ../HALO
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=~/mylibs/SEAL3.7 -DCMAKE_INSTALL_PREFIX=~/mylibs/SEAL3.7 -DCMAKE_BUILD_TYPE=Release
cmake --build . --target all -j
```

# A Guide For HEIR Experiment Evaluation (Original Version)
This artifact included three experiments for arithmetic-circuit programs, logic-circuit programs and end-to-end hybrid-circuit programs.
The concrete steps for evaluating these experiments are provided in ``benchmarks/arithmetic/README.md``, ``benchmarks/logic/README.md`` and ``benchmarks/e2e/README.md``.

# A Guide for Plots Generation
The concrete steps for evaluating these experiments are 
provided in ``HEIR_plot/README.md``.