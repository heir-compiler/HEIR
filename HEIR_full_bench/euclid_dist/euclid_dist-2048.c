#define N 2048
float euclid_dist(float a[N], float b[N])
{
    float result = (a[0] - b[0]) * (a[0] - b[0]);
    float temp;
    for (int i = 1; i < N; i++) {
        temp = a[i] - b[i];
        result += temp * temp;
    }
    return result;
}