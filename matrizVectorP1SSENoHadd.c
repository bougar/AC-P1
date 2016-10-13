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
	
	//Calculamos o múltiplo de 4 máis cercano de filas e de columnas
	//Para poder realizar os calculos con números que non sexan
	//múltiplo de catro
	n2 = n + (4 -(n % 4));
	m2 = m + (4 -(m % 4));


	//Reserva de memoria alineada a 16 bytes
    float *x = (float *) _mm_malloc(n2*sizeof(float), 16);
    float *A = (float *) _mm_malloc(m2*n2*sizeof(float), 16);
    float *y = (float *) _mm_malloc(m2*sizeof(float), 16);
	
	//Inicializamos a cero todos os vectores e matrices para 
	//evitar problemas de operación no algoritmo
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

	//Reserva de todos os rexistros utilizados no programa
	__m128 regA1, regA2, regA3, regA4;
	__m128 regAux1, regAux2, regAux3, regAux4;
	__m128 regAuxF1 , regAuxF2, regAuxF3, regAuxF4;
	__m128 regAlfa = _mm_set_ps(alfa, alfa, alfa, alfa);
	__m128 regAlfaX;
	__m128 regx,regy;
	__m128 regAdd, regAdd1, regAdd2;

    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m2 ; i+=4) {
		regy = _mm_load_ps(&(y[i]));
		regAdd = _mm_setzero_ps();
        for (j=0; (j<n2); j+=4) {
			
			//Cargamos o catro valores de rexistro X
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

			/*Pra sumar catro columnas dunha fila, necesitamos que cada columna se reparta
			a un rexistro cada unha, tendo en conta que teñen que ter a mesma posicion*/
			regAux1 = _mm_shuffle_ps(regA1, regA2, _MM_SHUFFLE(1,0,1,0));
			regAux2 = _mm_shuffle_ps(regA3, regA4, _MM_SHUFFLE(1,0,1,0));

			regAuxF1 = _mm_shuffle_ps(regAux1, regAux2, _MM_SHUFFLE(2,0,2,0));
			regAuxF2 = _mm_shuffle_ps(regAux1, regAux2, _MM_SHUFFLE(3,1,3,1));

			regAux1 = _mm_shuffle_ps(regA1, regA2, _MM_SHUFFLE(3,2,3,2));
			regAux2 = _mm_shuffle_ps(regA3, regA4, _MM_SHUFFLE(3,2,3,2));

			regAuxF3 = _mm_shuffle_ps(regAux1, regAux2, _MM_SHUFFLE(2,0,2,0));
			regAuxF4 = _mm_shuffle_ps(regAux1, regAux2, _MM_SHUFFLE(3,1,3,1));

			//Sumamos en vertical de dous en dous.
			regAdd1 = _mm_add_ps(regAuxF1, regAuxF2);
			regAdd2 = _mm_add_ps(regAuxF3, regAuxF4);

			//Os dous rexistros anteriores deben sumarse entre sí para obter
			//a suma das catro columnas. Por outro lado debemos de levar a conta
			//das operacións anteriores, que neste caso almacenanse en "regAdd"
			//ao que lle sumamos a suma mencionada previamente
			regAdd = _mm_add_ps(_mm_add_ps(regAdd1, regAdd2),regAdd);
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

