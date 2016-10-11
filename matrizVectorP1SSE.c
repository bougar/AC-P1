#include <assert.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pmmintrin.h>

int main( int argc, char *argv[] ) {

    int m, n, n2, m2, test, i, j;
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
	n2 = n + (4 -(n % 4));
	m2 = m + (4 -(m % 4));

    float *x = (float *) malloc(n2*sizeof(float));
    float *A = (float *) malloc(m2*n2*sizeof(float));
    float *y = (float *) malloc(m2*sizeof(float));
	memset(A, 0, n2*m2*sizeof(float));
	memset(x, 0, n2*sizeof(float));
	memset(y, 0, m2*sizeof(float));

    // Se inicializan la matriz y los vectores

    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            A[i*n2+j] = 1+i+j;
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
        for(i=0; i<m2; i++){
            for(j=0; j<n2; j++){
                printf("%f ", A[i*n2+j]);
            }
            printf("\n");
        }

        printf("\nVector x es...\n");
        for(i=0; i<n2; i++){
            printf("%f ", x[i]);
        }
        printf("\n");

        printf("\nVector y al principio es...\n");
        for(i=0; i<m2; i++){
            printf("%f ", y[i]);
        }
        printf("\n");
    }

    // Parte fundamental del programa
	__m128 regA1, regA2, regA3, regA4;
	__m128 regAlfa = _mm_set_ps(alfa, alfa, alfa, alfa);
	__m128 regAlfaX;
	__m128 regx,regy;
	__m128 regAdd, regAdd1, regAdd2;

    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m2 ; i+=4) {
		regy = _mm_load_ps(&(y[i]));
		regAdd = _mm_setzero_ps();
        for (j=0; (j<n2); j+=4) {
            //y[i] += alfa*A[i*n+j]*x[j];
			regx = _mm_load_ps(&(x[j]));
			//Cargamos os flotantes nos rexistros de 128 bits
			regA1 = _mm_load_ps(&A[i*n2+j]);
			regA2 = _mm_load_ps(&A[i*n2+j+n2]);
			regA3 = _mm_load_ps(&A[i*n2+j+2*n2]);
			regA4 = _mm_load_ps(&A[i*n2+j+3*n2]);

			//Multiplicamos os 4 valores almacenados en X por alfa
			regAlfaX = _mm_mul_ps(regx, regAlfa);

			//Multipicamos o anterior, float a float contra os flotantes de A cargados de 4 en 4
			regA1 = _mm_mul_ps(regAlfaX, regA1);
			regA2 = _mm_mul_ps(regAlfaX, regA2);
			regA3 = _mm_mul_ps(regAlfaX, regA3);
			regA4 = _mm_mul_ps(regAlfaX, regA4);

			//Sumanse en horizontal, de dous en dous os elementos de cada rexistro
			//E os resultados almacenanse nun rexistro de 128 bits
			regAdd1 = _mm_hadd_ps(regA1, regA2);
			regAdd2 = _mm_hadd_ps(regA3, regA4);

			//O obxetivo e sumar horizontalmente os catro elementos de cada rexistro
			//Dado que co as operacións anterior so sumamos de 2 en 2, necesitamos
			//outra operación de suma horizontal, para que finalmente nos devolva
			//un rexitro que conteña a suma horizontal de todolos elementos
			//de cada rexitro inicial. Por último levamos as contas nun rexistro
			//no que vamos facendo sumas ao resultado anterior para si acabar
			//coa suma das filas.
			regAdd = _mm_add_ps(_mm_hadd_ps(regAdd1, regAdd2),regAdd);
		}
		//Unha vez chegado o final de cada conxunto de bloques en horizontal,
		//teremos un rexistro que almacena un máximo de 4 flotantes, que representan
		//O valor de aplicar a fórmula a catro filas, polo que xa podemos almacenalo
		//na variable desexada
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
                testy[i] += alfa*A[i*n2+j]*x[j];
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

    _mm_free(x);
    _mm_free(y);
    _mm_free(A);
	
    return 0;
}

