float min_value(float a[4], float min_value)
{
    min_value = a[0];
    for (int i = 1; i < 4; i++) {
        if (min_value > a[i]) min_value = a[i];
    }

    return min_value;
}