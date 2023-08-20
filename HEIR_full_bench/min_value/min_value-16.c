#define N 16
float min_value(float a[N], float min_value)
{
    min_value = a[0];
    for (int i = 1; i < N; i++) {
        if (min_value > a[i]) min_value = a[i];
    }

    return min_value;
}