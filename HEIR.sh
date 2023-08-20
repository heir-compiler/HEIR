#!/bin/bash

# Author: Zian Zhao
# A script for evaluating benchmarks automatically

# Check the number of parameters
if [ $# -ne 1 ]; then
  echo "Please provide the path of the input file as the parameter!"
  exit 1
fi

# Get file path
filePath=$1

# Check whether the file exists
if [ ! -f "$filePath" ]; then
  echo "File does not exists!"
  exit 1
fi

# Extract file name
folderPath=$(dirname "$filePath")
fileName=$(basename "$filePath")
fileNameWithoutExtension=$(echo "$fileName" | cut -d. -f1)


./tools/cgeist  "$filePath"\
    -function=$fileNameWithoutExtension -S \
    -raise-scf-to-affine \
    --memref-fullrank -O0 \
    >> "${folderPath}/${fileNameWithoutExtension}.mlir"

./tools/heir-opt "${folderPath}/${fileNameWithoutExtension}.mlir" \
    --affine-loop-unroll="unroll-full unroll-num-reps=4" \
    --arith-slot \
    >> "${folderPath}/${fileNameWithoutExtension}_emitc.mlir"

./build/bin/emitc-translate "${folderPath}/${fileNameWithoutExtension}_emitc.mlir" \
    --mlir-to-cpp \
    >> "${folderPath}/${fileNameWithoutExtension}.cpp"

rm "${folderPath}/${fileNameWithoutExtension}.mlir"
rm "${folderPath}/${fileNameWithoutExtension}_emitc.mlir"