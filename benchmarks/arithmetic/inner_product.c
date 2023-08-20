float inner_product(float a[4], float b[4])
{
    float result;
    result = a[0] * b[0];
    for (int i = 1; i < 4; i++) {
        result += a[i] * b[i];
    }
    return result;
}