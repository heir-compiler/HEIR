#define N 8
float database(float data[N], float acc)
{
    for (int i = 0; i < N; i++) {
        if (data[i] > 5) {
            acc += data[i];
        }
    }
    return acc;
}