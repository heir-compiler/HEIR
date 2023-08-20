#define N 64
float inner_product(float a[N], float b[N])
{
    float result;
    result = a[0] * b[0];
    for (int i = 1; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}