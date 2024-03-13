#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef USE_GSL
#include <gsl/gsl_rng.h>
#endif

#define RNG_TYPE 1  // 0 for GSL, 1 for C rand

void generate_random(int n, int m) {
    int i, j;

    #if RNG_TYPE == 1
    FILE *file = fopen("random_numbers.txt", "w");
   double **matrix = (double **)malloc(m * sizeof(double *));
    for (i = 0; i < m; i++) {
        matrix[i] = (double *)malloc(m * sizeof(double));
    }

    // Llenar la matriz con números aleatorios
    srand(time(0));
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            matrix[i][j] = rand() % (n+1);
        }
    }

    // Imprimir la matriz en el archivo
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            fprintf(file, "%lf ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    free(matrix);
   

    #elif RNG_TYPE == 0
    continue;

    #endif
}

int main() {
    int n = 20;
    int m = 9;
    generate_random(n, m);
    
    return 0;
}