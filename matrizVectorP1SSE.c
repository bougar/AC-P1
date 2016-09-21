#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pmmintrin.h>

int main( int argc, char *argv[] ) {

    int m, n, test, i, j;
    float alfa;
    struct timeval t0, t1, t;

    // Parámetro 1 -> m
    // Parámetro 2 -> n
    // Parámetro 3 -> alfa
    // Parámetro 4 -> booleano que nos indica si se desea imprimir matrices y vectores de entrada y salida
    if(argc>3){
        m = atoi(argv[1]); //Number of Rows
        n = atoi(argv[2]); //Number of Columns
        alfa = atof(argv[3]);
        test = atoi(argv[4]);
    }
    else{
        printf("NUMERO DE PARAMETROS INCORRECTO\n");
        exit(0);
    }

    float *x = (float *) malloc(n*sizeof(float));
    float *A = (float *) malloc(m*n*sizeof(float));
    float *y = (float *) malloc(m*sizeof(float));

    // Se inicializan la matriz y los vectores

    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            A[i*n+j] = 1+i+j;
        }
    }

    for(i=0; i<n; i++){
        x[i] = (1+i);
    }

    for(i=0; i<m; i++){
        y[i] = (1-i);
    }

    if(test){
        printf("\nMatriz A es...\n");
        for(i=0; i<m; i++){
            for(j=0; j<n; j++){
                printf("%f ", A[i*n+j]);
            }
            printf("\n");
        }

        printf("\nVector x es...\n");
        for(i=0; i<n; i++){
            printf("%f ", x[i]);
        }
        printf("\n");

        printf("\nVector y al principio es...\n");
        for(i=0; i<m; i++){
            printf("%f ", y[i]);
        }
        printf("\n");
    }

    // Parte fundamental del programa
	__m128 regA1, regA2, regA3, regA4;
	__m128 regAlfa = _mm_set_ps(alfa, alfa, alfa, alfa);
	__m128 regx,regy;
	__m128 regAdd, regAdd1, regAdd2;

    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m; i+=4) {
		regy = _mm_load_ps(&(y[i]));
		regAdd = _mm_setzero_ps();
        for (j=0; j<n; j+=4) {
            //y[i] += alfa*A[i*n+j]*x[j];
			regx = _mm_load_ps(&(x[j]));
			regA1 = _mm_load_ps(&A[i*n+j]);
			regA2 = _mm_load_ps(&A[i*n+j+n]);
			regA3 = _mm_load_ps(&A[i*n+j+2*n]);
			regA4 = _mm_load_ps(&A[i*n+j+3*n]);
			regA1 = _mm_mul_ps(_mm_mul_ps(regx, regAlfa), regA1);
			regA2 = _mm_mul_ps(_mm_mul_ps(regx, regAlfa), regA2);
			regA3 = _mm_mul_ps(_mm_mul_ps(regx, regAlfa), regA3);
			regA4 = _mm_mul_ps(_mm_mul_ps(regx, regAlfa), regA4);
			regAdd1 = _mm_hadd_ps(regA1, regA2);
			regAdd2 = _mm_hadd_ps(regA3, regA4);
			regAdd = _mm_add_ps(_mm_hadd_ps(regAdd1, regAdd2),regAdd);
		}
		_mm_store_ps(&(y[i]),_mm_add_ps(regAdd,regy));
    }
	
    assert (gettimeofday (&t1, NULL) == 0);
    timersub(&t1, &t0, &t);

    if(test){
        printf("\nAl final vector y es...\n");
        for(i=0; i<m; i++){
            printf("%f ", y[i]);
        }
        printf("\n");

        float *testy = (float *) malloc(m*sizeof(float));
        for(i=0; i<m; i++){
            testy[i] = 1-i;
        }

        // Se calcula el producto sin ninguna vectorización
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                testy[i] += alfa*A[i*n+j]*x[j];
            }
        }

        int errores = 0;
        for(i=0; i<m; i++){
            if(testy[i] != y[i]){
                errores++;
                printf("\n Error en la posicion %d porque %f != %f", i, y[i], testy[i]);
            }
        }
        printf("\n%d errores en el producto matriz vector con dimensiones %dx%d\n", errores, m, n);
        free(testy);
    }

    printf ("Tiempo      = %ld:%ld(seg:mseg)\n", t.tv_sec, t.tv_usec/1000);

    free(x);
    free(y);
    free(A);
	
    return 0;
}




