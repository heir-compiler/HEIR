#define N 16
float sgn(float input)
{
    return input;
}

float min_index(float a[N], float min_value)
{
    min_value = a[0];
    for (int i = 1; i < N; i++) {
        if (min_value > a[i]) min_value = a[i];
    }

    float min_index[N];
    for (int i = 0; i < N; i++) {
        min_index[i] = a[i] - min_value;
        min_index[i] = sgn(min_index[i]);
    }

    return min_value;
}