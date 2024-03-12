#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef USE_GSL
#include <gsl/gsl_rng.h>
#endif

#define RNG_TYPE 1  // 0 for GSL, 1 for C rand

int generate_random(int n, int m) {
    int result = 0;
    int i=0;

    #if RNG_TYPE == 0
    #ifdef USE_GSL
    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    fopen("random_numbers.txt", "w", stdout);
    do {
        result = gsl_rng_uniform_int(r, n+1);
        fprintf(file, "%d\n", result);
        i++;
    } while (i < m);

    gsl_rng_free (r);
    fclose(file);

    #elif RNG_TYPE == 1
    FILE *file = fopen("random_numbers.txt", "w");
    srand(time(0));
        do
        {   
            result = rand() % (n+1);
            fprintf(file, "%d\n", result);
             i++;
        }while (i<m);
        fclose(file);
   
    #endif
}

int main() {
    int n = 100;
    int m = 300;
    int random_number = generate_random(n, m);
    return 0;
}