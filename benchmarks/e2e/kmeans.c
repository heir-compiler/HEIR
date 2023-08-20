#define DATANUM 5
#define DATADIM 3
#define CENTNUM 2

float euclid_dist(float data[DATADIM], float cent[DATADIM])
{
    return data[0];
}

float lut_abs(float a)
{
    return a;
}

float lut_lsz(float a)
{
    return a;
}

float inner_product(float a[DATADIM], float b[DATADIM])
{
    return a[0];
}

float lut_inverse(float a)
{
    return a;
}

float kmeans(float data[DATANUM][DATADIM], float cent[CENTNUM][DATADIM])
{
    float data_tran[DATADIM][DATANUM];
    for (int i = 0; i < DATANUM; i++) {
        for (int j = 0; j < DATADIM; j++) {
            data_tran[j][i] = data[i][j];
        }
    }
    
    float dist_matrix[DATANUM][CENTNUM];
    for (int i = 0; i < DATANUM; i++) {
        for (int j = 0; j < CENTNUM; j++) {
            dist_matrix[i][j] = euclid_dist(data[i], cent[j]);
        }
    }

    // Compute Min-Index
    //  1. Compute Min_value
    //  2. Mi = Mi - min
    //  3. LUT(Mi < 0)
    float index_matrix[DATANUM][CENTNUM];
    float min_value;
    float sub;
    // float add;
    float msb;
    for (int i = 0; i < DATANUM; i++) {
        // Compute min_value
        sub = dist_matrix[i][0] - dist_matrix[i][1];
        msb = lut_lsz(sub);
        min_value = msb * dist_matrix[i][0] +(1 - msb) * dist_matrix[i][1];
        // Mi = Mi - min_value
        index_matrix[i][0] = dist_matrix[i][0] + dist_matrix[i][0];
        index_matrix[i][0] = index_matrix[i][0] - min_value;
        index_matrix[i][1] = dist_matrix[i][1] + dist_matrix[i][1];
        index_matrix[i][1] = index_matrix[i][1] - min_value;
        // LUT(Mi < 0)
        index_matrix[i][0] = lut_lsz(index_matrix[i][0]);
        index_matrix[i][1] = lut_lsz(index_matrix[i][1]);
    }

    float index_column[CENTNUM][DATANUM];
    for (int i = 0; i < DATANUM; i++) {
        for (int j = 0; j < CENTNUM; j++) {
            index_column[j][i] = index_matrix[i][j];
        }
    }

    float cent_result[CENTNUM][DATADIM];
    float inverse;
    for(int i = 0; i < CENTNUM; i++) {
        inverse = index_matrix[0][i];
        for (int k = 1; k < DATANUM; k++) {
            inverse = inverse + index_matrix[k][i];
        }
        inverse = lut_inverse(inverse);
        for (int j = 0; j < DATADIM; j++) {
            cent_result[i][j] = inner_product(data_tran[j], index_column[i]);
            cent_result[i][j] = inverse * cent_result[i][j];
        }
    }

    return cent_result[1][1];
    // return inverse;

}