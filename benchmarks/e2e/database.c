float database(float data[4], float acc)
{
    for (int i = 0; i < 4; i++) {
        if (data[i] > 5) {
            acc += data[i];
        }
    }
    return acc;
}