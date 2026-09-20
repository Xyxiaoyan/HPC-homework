#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 定义矩阵大小，默认值为 10
#ifndef SIZE
#define SIZE 10
#endif

// 定义重复次数，默认值为 10000
#ifndef REPEAT
#define REPEAT 100000
#endif

// 定义矩阵 A、X、Y 和 Z
static double A[SIZE][SIZE];
static double X[SIZE][SIZE];
static double Y[SIZE][SIZE];
static double Z[SIZE][SIZE];

// 产生随机数
double random_double(void)
{
    return 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

// 初始化矩阵 A、X、Y 和 Z
void initialize(void)
{
    int i, j;

    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {

            /* A is a constant matrix */
            A[i][j] = 2.0;

            /* X and Y contain random floating-point values */
            X[i][j] = random_double();
            Y[i][j] = random_double();

            Z[i][j] = 0.0;
        }
    }
}

// 将 Y 复制到 Z
void reset_Z(void)
{
    memcpy(Z, Y, sizeof(Y));
}

// 定义不同的循环顺序方法来计算矩阵乘法
void method_ijk(void)
{
    int i, j, k;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            for (k = 0; k < SIZE; k++) {
                Z[i][j] += A[i][k] * X[k][j];
            }
        }
    }
}

void method_kij(void)
{
    int i, j, k;
    for (k = 0; k < SIZE; k++) {
        for (i = 0; i < SIZE; i++) {
            for (j = 0; j < SIZE; j++) {
                Z[i][j] += A[i][k] * X[k][j];
            }
        }
    }
}

void method_jki(void)
{
    int i, j, k;
    for (j = 0; j < SIZE; j++) {
        for (k = 0; k < SIZE; k++) {
            for (i = 0; i < SIZE; i++) {
                Z[i][j] += A[i][k] * X[k][j];
            }
        }
    }
}

// 题目a
void daxpy(void)
{
    reset_Z();
    method_ijk();
}

void print_results(void)
{
    int i, j;
    printf("\nPart (a): Matrix calculation results\n");
    printf("Z = A * X + Y\n\n");
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            printf(
                "i=%d, j=%d, A=%.2f, X=%.2f, Y=%.2f, Z=%.2f\n",
                i,
                j,
                A[i][j],
                X[i][j],
                Y[i][j],
                Z[i][j]
            );
        }
    }
}

// 计算各个值是否正确
void manual_check(void)
{
    int k;
    double value;
    value = Y[0][0];
    for (k = 0; k < SIZE; k++) {
        value += A[0][k] * X[k][0];
    }
    printf("\nManual check for Z[0][0]:\n");
    printf("Calculated by hand formula = %.6f\n", value);
    printf("Program result             = %.6f\n", Z[0][0]);
}

// 通过计算矩阵 Z 的所有元素的和来计算校验和
double checksum(void)
{
    double sum = 0.0;
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            sum += Z[i][j];
        }
    }

    return sum;
}

int main(void)
{
    int r;
    clock_t start;
    clock_t end;
    double total_time;
    double average_time;
    double daxpy_total;
    double daxpy_average;
    double ijk_total;
    double kij_total;
    double jki_total;
    double ijk_average;
    double kij_average;
    double jki_average;

    volatile double check_ijk;
    volatile double check_kij;
    volatile double check_jki;

    printf("========================================\n");
    printf("SIZE   = %d\n", SIZE);
    printf("REPEAT = %d\n", REPEAT);
    printf("========================================\n");
    // 初始化矩阵
    initialize();
    // 计算题目a
    daxpy();
    // 打印结果和检验结果
    print_results();
    manual_check();
    // 题目c
    printf("\n========================================\n");
    printf("Part (c): Loop order comparison\n");
    printf("========================================\n");
    
    // 预热三种方法，确保CPU进入工作状态
    for (r = 0; r < REPEAT; r++) {
        reset_Z();
        method_ijk();
        reset_Z();
        method_kij();
        reset_Z();
        method_jki();
    }

    start = clock();
    for (r = 0; r < REPEAT; r++) {
        reset_Z();
        method_ijk();
    }
    end = clock();
    ijk_total =
        (double)(end - start) / CLOCKS_PER_SEC;
    ijk_average =
        ijk_total / REPEAT;
    check_ijk = checksum();

    start = clock();
    for (r = 0; r < REPEAT; r++) {
        reset_Z();
        method_kij();
    }
    end = clock();
    kij_total =
        (double)(end - start) / CLOCKS_PER_SEC;
    kij_average =
        kij_total / REPEAT;
    check_kij = checksum();

    start = clock();
    for (r = 0; r < REPEAT; r++) {
        reset_Z();
        method_jki();
    }
    end = clock();
    jki_total =
        (double)(end - start) / CLOCKS_PER_SEC;
    jki_average =
        jki_total / REPEAT;
    check_jki = checksum();

    printf(
        "ijk: total = %.9f s, average = %.9f s, checksum = %.4f\n",
        ijk_total,
        ijk_average,
        check_ijk
    );
    printf(
        "kij: total = %.9f s, average = %.9f s, checksum = %.4f\n",
        kij_total,
        kij_average,
        check_kij
    );
    printf(
        "jki: total = %.9f s, average = %.9f s, checksum = %.4f\n",
        jki_total,
        jki_average,
        check_jki
    );
    return 0;
}