#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s N THREADS\n", argv[0]);
        return 1;
    }
    size_t n = strtoull(argv[1], NULL, 10);
    int threads = atoi(argv[2]);
    if (!n || threads <= 0)
        return 1;
    double *a = malloc(n * n * sizeof(double)), *b = malloc(n * n * sizeof(double)), *c = calloc(n * n, sizeof(double));
    if (!a || !b || !c)
    {
        perror("allocation");
        return 1;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
        {
            a[i * n + j] = (double)((i + j) % 17 + 1) / 17.;
            b[i * n + j] = (double)((i * 3 + j) % 19 + 1) / 19.;
        }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0;
            for (size_t k = 0; k < n; k++)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
    double seconds = omp_get_wtime() - start, checksum = 0;
    for (size_t i = 0; i < n * n; i++)
        checksum += c[i];
    // Validate a few entries independently (outside timing).
    int ok = isfinite(checksum);
    for (size_t i = 0; i < n; i += n / 4 + 1)
        for (size_t j = 0; j < n; j += n / 4 + 1)
        {
            double ref = 0;
            for (size_t k = 0; k < n; k++)
                ref += a[i * n + k] * b[k * n + j];
            if (fabs(ref - c[i * n + j]) > 1e-10 * fmax(1.0, fabs(ref)))
                ok = 0;
        }
    printf("matrix,%zu,1,%d,%.9f,%.12g,%s\n", n, threads, seconds, checksum, ok ? "PASS" : "FAIL");
    free(a);
    free(b);
    free(c);
    return ok ? 0 : 2;
}
