#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s N LOOPS THREADS\n", argv[0]);
        return 1;
    }
    size_t n = strtoull(argv[1], NULL, 10);
    long loops = strtol(argv[2], NULL, 10);
    int threads = atoi(argv[3]);
    if (!n || loops <= 0 || threads <= 0)
        return 1;
    double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
    if (!x || !y)
    {
        perror("malloc");
        return 1;
    }
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1.0 + (i % 13) * 0.01;
        y[i] = 0.5 + (i % 7) * 0.01;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    const double a = 0.000001;
    double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
    {
        double v = y[i], increment = a * x[i];
        for (long r = 0; r < loops; r++)
            v += increment;
        y[i] = v;
    }
    double seconds = omp_get_wtime() - start, checksum = 0, expected = 0;
    for (size_t i = 0; i < n; i++)
    {
        checksum += y[i];
        expected += 0.5 + (i % 7) * 0.01 + (double)loops * a * x[i];
    }
    double rel = fabs(checksum - expected) / fmax(1.0, fabs(expected));
    printf("daxpy,%zu,%ld,%d,%.9f,%.12g,%.3g,%s\n", n, loops, threads, seconds, checksum, rel, rel < 1e-6 ? "PASS" : "FAIL");
    free(x);
    free(y);
    return rel < 1e-6 ? 0 : 2;
}
