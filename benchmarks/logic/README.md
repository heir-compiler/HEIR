# Experment (E2):  Logic Experiments
## 2.1 Min Value Evaluation
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/logic/min_value.c \
    -function=min_value -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/logic/min_value.mlir
./tools/heir-opt ./benchmarks/logic/min_value.mlir \
    --affine-loop-unroll="unroll-full=true" \
    --logic-emitc \
    >> ./benchmarks/logic/min_value_emitc.mlir
./tools/emitc-translate ./benchmarks/logic/min_value_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/logic/min_value.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/logic/min_value.cpp -o ../HALO/ndss_experiment/min_value.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/min_value
```
## 2.2 Min Index Evaluation
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/logic/min_index.c \
    -function=min_index -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/logic/min_index.mlir
./tools/heir-opt ./benchmarks/logic/min_index.mlir \
    --affine-loop-unroll="unroll-full=true" \
    --logic-emitc \
    >> ./benchmarks/logic/min_index_emitc.mlir
./tools/emitc-translate ./benchmarks/logic/min_index_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/logic/min_index.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/logic/min_index.cpp -o ../HALO/ndss_experiment/min_index.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/min_index
```

## 2.3 Fibonacci Evaluation
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/logic/fibonacci.c \
    -function=fibonacci -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/logic/fibonacci.mlir
./tools/heir-opt ./benchmarks/logic/fibonacci.mlir \
    --branch --affine-loop-unroll="unroll-full=true" \
    --logic-emitc \
    >> ./benchmarks/logic/fibonacci_emitc.mlir
./tools/emitc-translate ./benchmarks/logic/fibonacci_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/logic/fibonacci.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/logic/fibonacci.cpp -o ../HALO/ndss_experiment/fibonacci.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/fibonacci
```