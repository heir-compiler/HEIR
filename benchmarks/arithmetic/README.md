<!-- # HEIR Experiment Evaluation -->
# Experment (E1): Arithmetic Experiments
## 1.1 Inner Product Evaluation.
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/arithmetic/inner_product.c \
    -function=inner_product -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/arithmetic/inner_product.mlir
./tools/heir-opt ./benchmarks/arithmetic/inner_product.mlir \
    --affine-loop-unroll="unroll-full=true" \
    --arith-emitc \
    >> ./benchmarks/arithmetic/inner_product_emitc.mlir
./tools/emitc-translate ./benchmarks/arithmetic/inner_product_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/arithmetic/inner_product.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/arithmetic/inner_product.cpp -o ../HALO/ndss_experiment/inner_product.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/inner_product
```

## 1.2 Euclidean distance evaluation. 
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/arithmetic/euclid_dist.c \
    -function=euclid_dist -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/arithmetic/euclid_dist.mlir
./tools/heir-opt ./benchmarks/arithmetic/euclid_dist.mlir \
    --affine-loop-unroll="unroll-full=true" \
    --arith-emitc \
    >> ./benchmarks/arithmetic/euclid_dist_emitc.mlir
./tools/emitc-translate ./benchmarks/arithmetic/euclid_dist_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/arithmetic/euclid_dist.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/arithmetic/euclid_dist.cpp -o ../HALO/ndss_experiment/euclid_dist.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/euclid_dist
```