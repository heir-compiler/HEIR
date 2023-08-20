#!/bin/bash

CURRENT_DIR=$(cd $(dirname $0); pwd)

# EVA Results
python /home/NDSS_Artifact/EVA/examples/euclid_dist/euclid_dist_16.py >> $CURRENT_DIR/eucliddist.out
python /home/NDSS_Artifact/EVA/examples/euclid_dist/euclid_dist_64.py >> $CURRENT_DIR/eucliddist.out
python /home/NDSS_Artifact/EVA/examples/euclid_dist/euclid_dist_256.py >> $CURRENT_DIR/eucliddist.out
python /home/NDSS_Artifact/EVA/examples/euclid_dist/euclid_dist_512.py >> $CURRENT_DIR/eucliddist.out
python /home/NDSS_Artifact/EVA/examples/euclid_dist/euclid_dist_2048.py >> $CURRENT_DIR/eucliddist.out
python /home/NDSS_Artifact/EVA/examples/euclid_dist/euclid_dist_4096.py >> $CURRENT_DIR/eucliddist.out

echo "EVA evaluation done"

# HECO Results
/home/NDSS_Artifact/SEAL/build/bin/heco_eucliddist_16 >> $CURRENT_DIR/eucliddist.out
/home/NDSS_Artifact/SEAL/build/bin/heco_eucliddist_64 >> $CURRENT_DIR/eucliddist.out
/home/NDSS_Artifact/SEAL/build/bin/heco_eucliddist_256 >> $CURRENT_DIR/eucliddist.out
/home/NDSS_Artifact/SEAL/build/bin/heco_eucliddist_512 >> $CURRENT_DIR/eucliddist.out
/home/NDSS_Artifact/SEAL/build/bin/heco_eucliddist_2048 >> $CURRENT_DIR/eucliddist.out
/home/NDSS_Artifact/SEAL/build/bin/heco_eucliddist_4096 >> $CURRENT_DIR/eucliddist.out

echo "HECO evaluation done"

# HEIR Results
fileList=(
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/euclid_dist/euclid_dist-16.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/euclid_dist/euclid_dist-64.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/euclid_dist/euclid_dist-256.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/euclid_dist/euclid_dist-512.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/euclid_dist/euclid_dist-2048.c"
    "/home/NDSS_Artifact/HEIR/HEIR_full_bench/euclid_dist/euclid_dist-4096.c"
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
        --affine-loop-unroll="unroll-full unroll-num-reps=4" \
        --arith-emitc \
        >> "${folderPath}/${fileNameWithoutExtension}_emitc.mlir"

    $CURRENT_DIR/../../tools/emitc-translate "${folderPath}/${fileNameWithoutExtension}_emitc.mlir" \
        --mlir-to-cpp \
        >> "${folderPath}/${fileNameWithoutExtension}.cpp"


    python "$CURRENT_DIR/../../format_assistant/halo_transmitter.py" \
        -i "${folderPath}/${fileNameWithoutExtension}.cpp" \
        -o "/home/NDSS_Artifact/HALO/ndss_plot/euclid_dist/${fileNameWithoutExtension}.cpp"

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

    ./build/bin/"heir_${fileNameWithoutExtension}" >> $CURRENT_DIR/eucliddist.out
done

echo "HEIR evaluation done"