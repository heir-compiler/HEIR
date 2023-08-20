float euclid_dist(float a[4], float b[4])
{
    float result = (a[0] - b[0]) * (a[0] - b[0]);
    float temp;
    for (int i = 1; i < 4; i++) {
        temp = a[i] - b[i];
        result += temp * temp;
    }
    return result;
}