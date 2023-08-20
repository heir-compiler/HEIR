# A Guide for Plots Generation
In this directory, scripts are provided for the purpose of 
comparing with other FHE compilers and generating plots.

## 1. Compiling Benchmarks with other FHE compilers 
As HECO still faces challenges in integrating its IR code with the 
underlying FHE library, we manually implemented each benchmark on 
SEAL library using the generated IR. The source code for these 
benchmarks can be found in `SEAL/native/HECOBench`. Please follow 
the instructions in `HEIR/README.md` to bulld SEAL first for 
generating the executables of these benchmarks.  


The benchmark programs for the EVA compiler are located in 
`EVA/examples`. EVA can evaluate these benchmarks by executing 
the corresponding Python scripts.

The benchmark programs for the Transpiler are located in 
`Transpiler/transpiler/examples`. To evaluate the benchmarks, 
you can execute the following instruction: 
```sh
# Take min_value (vector size = 2) program as an example
cd Transpiler/
bazel run //transpiler/examples/my_min_value/min_value2:min_value_tfhe_testbench
```

**Note: Instead of evaluating the benchmarks one by one, please utilize 
the scripts provided in the following section to compare the 
performance among different FHE compilers.**

## 2. Generating Plots
**Note that the complete input prgrams for HEIR benchmarks are 
located in `HEIR/HEIR_full_bench`.  

Before evaluating these benchmarks, please use the following 
instructions to build HALO and SEAL:
```sh
# Build SEAL
cd /home/NDSS_Artifact/SEAL/
cmake -S . -B build -DSEAL_USE_INTEL_HEXL=ON
cmake --build build -j

# Build HALO
cd /home/NDSS_Artifact/HALO/
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/home/NDSS_Artifact/SEAL/build/ \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --target all -j
```

### 2.1. Arithmetic Circuit Evaluation
To generate the plots for runtime latency comparison in evaluating 
InnerProduct, the runtime latency of different FHE compilers needs 
to be evaluated and logged in `innerproduct.out`.
```sh
./InnerProduct/GenInnerProductResults.sh
```
Next, generate the plot depicting the runtime latency evaluated in 
the previous step.
```sh
python ./InnerProduct/plot_inner.py
```

Similarly, to generate plots for the runtime latency comparison in 
evaluating EuclidDist, please execute the following instructions.
```sh
./EuclidDist/GenEuclidDistResults.sh
python ./EuclidDist/plot_euclid.py
```
### 2.2. Logic Circuit Evaluation
To generate the plots for the MinValue, MinIndex, and Fibonacci 
benchmarks, please follow the instructions below.
```sh
./MinValue/GenMinValueResults.sh full
python ./MinValue/plot_value.py full

./MinIndex/GenMinIndexResults.sh full
python ./MinIndex/plot_index.py full

./Fibonacci/GenFibonacciResults.sh full
python ./Fibonacci/plot_fibonacci.py full
```
However, generating the full version of experiment results may take 
hours. To generate a shortened version of the comparison result, please 
follow the instructions below.
```sh
./MinValue/GenMinValueResults.sh short
python ./MinValue/plot_value.py short

./MinIndex/GenMinIndexResults.sh short
python ./MinIndex/plot_index.py short

./Fibonacci/GenFibonacciResults.sh short
python ./Fibonacci/plot_fibonacci.py short
```
### 2.3 Database evaluation
Similarly, due to the significant runtime latency, only a condensed version
of the comparison result is provied. Please refer to the instructions
below.
```sh
./Database/GenDatabaseResults.sh
python ./Database/plot_database.py
```