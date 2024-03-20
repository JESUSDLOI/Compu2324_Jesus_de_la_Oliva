#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void mtrz_aleatoria(int N) {
    int i, j;
    int** matriz = (int**)malloc(N * sizeof(int*));
    for (i = 0; i < N; i++) {
        matriz[i] = (int*)malloc(N * sizeof(int));
        for (j = 0; j < N; j++) {
            matriz[i][j] = rand() % N;
            printf("%d ", matriz[i][j]);
        }
        printf("\n");
    }
    for (i = 0; i < N; i++) {
        free(matriz[i]);
    }
    free(matriz);
}

int main() {
    srand(time(0));  // Para generar números aleatorios diferentes en cada ejecución
    clock_t start_time, end_time;
    double execution_time;

    start_time = clock();

    mtrz_aleatoria(10000);

    end_time = clock();
    execution_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("El programa se ejecutó en: %f segundos\n", execution_time);

    return 0;
}