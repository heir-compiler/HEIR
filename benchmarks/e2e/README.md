# Experiment (E3): End-To-End Applications
## 3.1 Database Filter-Aggregate Evaluation
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/e2e/database.c \
    -function=database -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/e2e/database.mlir
./tools/heir-opt ./benchmarks/e2e/database.mlir \
    --branch --affine-loop-unroll="unroll-full=true" \
    --logic-emitc \
    >> ./benchmarks/e2e/database_emitc.mlir
./tools/emitc-translate ./benchmarks/e2e/database_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/e2e/database.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/e2e/database.cpp -o ../HALO/ndss_experiment/database.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/database
```

## 3.2 K-Means Evaluation
Execute the following instructions in ``HEIR/`` directory.

First, we compile the original C program to C++ homomorphic APIs and plug it to HALO library.
```sh
./tools/cgeist ./benchmarks/e2e/kmeans.c \
    -function=kmeans -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> ./benchmarks/e2e/kmeans.mlir
python ./format_assistant/polygeist_eliminator.py -i ./benchmarks/e2e/kmeans.mlir
./tools/heir-opt ./benchmarks/e2e/kmeans.mlir \
    --branch --affine-loop-unroll="unroll-full unroll-num-reps=4" \
    --logic-emitc \
    >> ./benchmarks/e2e/kmeans_emitc.mlir
./tools/emitc-translate ./benchmarks/e2e/kmeans_emitc.mlir \
    --mlir-to-cpp \
    >> ./benchmarks/e2e/kmeans.cpp
python ./format_assistant/halo_transmitter.py -i ./benchmarks/e2e/kmeans.cpp -o ../HALO/ndss_experiment/kmeans.cpp
```
Then, we execute the generated benchmark.
```sh
cd ../HALO
cmake --build ./build --config Release --target all
./build/bin/kmeans
```