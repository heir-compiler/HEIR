#!/bin/bash

CURRENT_DIR=$(cd $(dirname $0); pwd)

# Transpiler Results
cd /home/NDSS_Artifact/Transpiler
bazel run //transpiler/examples/my_database/database2:database_tfhe_testbench >> $CURRENT_DIR/database.out
bazel run //transpiler/examples/my_database/database4:database_tfhe_testbench >> $CURRENT_DIR/database.out
bazel run //transpiler/examples/my_database/database8:database_tfhe_testbench >> $CURRENT_DIR/database.out
# bazel run //transpiler/examples/my_database/database16:database_tfhe_testbench >> $CURRENT_DIR/database.out
# bazel run //transpiler/examples/my_database/database32:database_tfhe_testbench >> $CURRENT_DIR/database.out

echo "Transpiler Evaluation done"

# HEIR Results
fileList=(
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/database/database-2.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/database/database-4.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/database/database-8.c"
    # "/home/NDSS_Artifact/HEIR/HEIR_full_bench/database/database-16.c"
    # "/home/NDSS_Artifact/HEIR/HEIR_full_bench/database/database-32.c"
)

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
        -o "/home/NDSS_Artifact/HALO/ndss_plot/database/${fileNameWithoutExtension}.cpp"

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

    ./build/bin/"heir_${fileNameWithoutExtension}" >> $CURRENT_DIR/database.out
done

echo "HEIR evaluation done"