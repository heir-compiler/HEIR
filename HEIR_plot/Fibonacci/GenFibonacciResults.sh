#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Select to generate 'short' or 'full' version of experiment results"
  exit 1
fi

CURRENT_DIR=$(cd $(dirname $0); pwd)

# Transpiler Results
cd /home/NDSS_Artifact/Transpiler
bazel run //transpiler/examples/my_fibonacci/fibonacci4:fibonacci_tfhe_testbench >> $CURRENT_DIR/fibonacci.out
bazel run //transpiler/examples/my_fibonacci/fibonacci8:fibonacci_tfhe_testbench >> $CURRENT_DIR/fibonacci.out
bazel run //transpiler/examples/my_fibonacci/fibonacci16:fibonacci_tfhe_testbench >> $CURRENT_DIR/fibonacci.out
if [ "$1" == "full" ]; then
    bazel run //transpiler/examples/my_fibonacci/fibonacci32:fibonacci_tfhe_testbench >> $CURRENT_DIR/fibonacci.out
    bazel run //transpiler/examples/my_fibonacci/fibonacci64:fibonacci_tfhe_testbench >> $CURRENT_DIR/fibonacci.out
fi

echo "Transpiler Evaluation done"

# HEIR Results
fileList=(
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-4.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-8.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-16.c"
)

if [ "$1" == "full" ]; then
    fileList=(
        "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-4.c"
        "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-8.c"
        "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-16.c"
        "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-32.c"
        "/home/NDSS_Artifact/HEIR/HEIR_full_bench/fibonacci/fibonacci-64.c"
    )
fi

for filePath in "${fileList[@]}"; do
    # Extract file name
    folderPath=$(dirname "$filePath")
    fileName=$(basename "$filePath")
    fileNameWithoutExtension=$(echo "$fileName" | cut -d. -f1)
    functionName=$(echo "$fileNameWithoutExtension" | cut -d- -f1)

    $CURRENT_DIR/../../tools/cgeist  "$filePath"\
        -function=$functionName -S \
        -raise-scf-to-affine \
        --memref-fullrank -O0 \
        >> "${folderPath}/${fileNameWithoutExtension}.mlir"

    $CURRENT_DIR/../../tools/heir-opt "${folderPath}/${fileNameWithoutExtension}.mlir" \
        --branch --affine-loop-unroll="unroll-full unroll-num-reps=4" \
        --logic-emitc \
        >> "${folderPath}/${fileNameWithoutExtension}_emitc.mlir"

    $CURRENT_DIR/../../tools/emitc-translate "${folderPath}/${fileNameWithoutExtension}_emitc.mlir" \
        --mlir-to-cpp \
        >> "${folderPath}/${fileNameWithoutExtension}.cpp"


    python "$CURRENT_DIR/../../format_assistant/halo_transmitter.py" \
        -i "${folderPath}/${fileNameWithoutExtension}.cpp" \
        -o "/home/NDSS_Artifact/HALO/ndss_plot/fibonacci/${fileNameWithoutExtension}.cpp"

    rm "${folderPath}/${fileNameWithoutExtension}.mlir"
    rm "${folderPath}/${fileNameWithoutExtension}_emitc.mlir"
    rm "${folderPath}/${fileNameWithoutExtension}.cpp"
done

cd /home/NDSS_Artifact/HALO
cmake --build ./build --target all -j

for filePath in "${fileList[@]}"; do
    folderPath=$(dirname "$filePath")
    fileName=$(basename "$filePath")
    fileNameWithoutExtension=$(echo "$fileName" | cut -d. -f1)

    ./build/bin/"heir_${fileNameWithoutExtension}" >> $CURRENT_DIR/fibonacci.out
done

echo "HEIR evaluation done"