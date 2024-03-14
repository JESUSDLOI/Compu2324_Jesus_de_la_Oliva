#include <stdio.h>
#include <stdlib.h>

void suma_vectores(int *v1, int *v2, int *v3, int n){
    int i;
    for(i=0; i<n; i++){
        v3[i] = v1[i] + v2[i];
    }
}

int main () {
    int n = 10;
    int v1[n], v2[n], v3[n];
    int i;
    for(i=0; i<n; i++){
        v1[i] = i;
        v2[i] = i;
    }
suma_vectores(v1, v2, v3, n);

    for(i=0; i<n; i++){

        printf("%d ", v3[i]);
    }
    return 0;
}