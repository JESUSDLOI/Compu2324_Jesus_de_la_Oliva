#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gsl_rng.h"

// Para compilar en Joel gcc -o Programa Programa.c `gsl-config --cflags --libs`

#define RNG_TYPE 0  // 0 for GSL, 1 for C rand

gsl_rng *tau;

void generate_random(int n, int m) {
    int i, j;
    extern gsl_rng *tau;
    int semilla=18237247; 
    //Una vez en el programa tenemos que incializar el generador. Para eso le tenemos
    //que pasar un número entero, de al menos 5 dígitos, que se usará de semilla

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

    int *r=NULL;

    tau=gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(tau,semilla);

    //Generar un número aleatorio
    for(i=0;i<m;i++){
        r[i] =gsl_rng_uniform_int(tau,n);
    }
    free(r);
    gsl_rng_free(tau);
    
    FILE *file = fopen("random_numbers_joel.txt", "w");
    for (i = 0; i < m; i++)
        fprintf(file, "%d ", r[i]);
    
    fclose(file);

    #endif
}

int main() {
    int n = 20;
    int m = 9;
    generate_random(n, m);
    
    return 0;
}