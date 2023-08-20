float sgn(float input)
{
    return input;
}

float min_index(float a[4], float min_value)
{
    min_value = a[0];
    for (int i = 1; i < 4; i++) {
        if (min_value > a[i]) min_value = a[i];
    }

    float min_index[4];
    for (int i = 0; i < 4; i++) {
        min_index[i] = a[i] - min_value;
        min_index[i] = sgn(min_index[i]);
    }

    return min_value;
}