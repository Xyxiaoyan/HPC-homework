#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double *alloc_vec(size_t n)
{
    double *p = malloc(n * sizeof(double));
    if (!p)
    {
        perror("malloc");
        exit(1);
    }
    return p;
}

static double checksum(const double *y, size_t n)
{
    double s = 0;
    for (size_t i = 0; i < n; i++)
        s += y[i];
    return s;
}

static void init(double *x, double *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1.0 + (i % 13) * 0.01;
        y[i] = 0.5 + (i % 7) * 0.01;
    }
}

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
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    double *x = alloc_vec(n), *y = alloc_vec(n);
    const double a = 0.000001;
    double times[3], sums[3];
    for (int mode = 0; mode < 3; mode++)
    {
        init(x, y, n);
        double start = omp_get_wtime();
        if (mode == 0)
        {
            for (long r = 0; r < loops; r++)
                for (size_t i = 0; i < n; i++)
                    y[i] += a * x[i];
        }
        else if (mode == 1)
        {
            // Parallelize repetitions with private accumulators, then merge.
            // A flat allocation avoids a non-standard variable length 2D array.
            double *partial = calloc((size_t)threads * n, sizeof(double));
            if (!partial)
            {
                perror("calloc");
                return 1;
            }
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double *local = partial + (size_t)tid * n;
#pragma omp for schedule(static)
                for (long r = 0; r < loops; r++)
                    for (size_t i = 0; i < n; i++)
                        local[i] += a * x[i];
            }
            for (size_t i = 0; i < n; i++)
                for (int t = 0; t < threads; t++)
                    y[i] += partial[(size_t)t * n + i];
            free(partial);
        }
        else
        {
#pragma omp parallel
            {
                for (long r = 0; r < loops; r++)
                {
#pragma omp for schedule(static)
                    for (size_t i = 0; i < n; i++)
                        y[i] += a * x[i];
                }
            }
        }
        times[mode] = omp_get_wtime() - start;
        sums[mode] = checksum(y, n);
    }
    double expected = 0;
    for (size_t i = 0; i < n; i++)
        expected += (0.5 + (i % 7) * 0.01) + (double)loops * a * x[i];
    for (int mode = 0; mode < 3; mode++)
    {
        double rel = fabs(sums[mode] - expected) / fmax(1.0, fabs(expected));
        fprintf(stdout, "part1,%zu,%ld,%d,%d,%.9f,%.12g,%.3g,%s\n", n, loops, mode == 0 ? 1 : threads, mode + 1, times[mode], sums[mode], rel, rel < 1e-6 ? "PASS" : "FAIL");
        if (rel >= 1e-6)
            return 2;
    }
    free(x);
    free(y);
    return 0;
}
