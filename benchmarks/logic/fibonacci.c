float fibonacci(float pre, float cur, float count, float output)
{
    float temp;
    for (int i = 0; i < 4; i++) {
        if (i < count) {
            temp = cur;
            cur = pre + cur;
            pre = temp;
        }
        output = cur;
    }
    return output;
}