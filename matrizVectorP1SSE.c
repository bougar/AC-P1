#include <assert.h>
#include <string.h> 
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pmmintrin.h>
#include <malloc.h>
__m128 fillRegister(int i, int j, int n, float * data){
	static float toRegister[4];
	memset(toRegister, 0, sizeof(float) * 4);
	int c,d;
	for ( d=i; d < i + 4; d++ ){
		for ( c=j; c < n && (c-j) < 4; c++ ){
			toRegister[c-j]=data[d*n+c];
		}
		return _mm_load_ps(toRegister);
	}
}

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

    float *x = (float *) _mm_malloc(n*sizeof(float), 16);
    float *A = (float *) _mm_malloc(m*n*sizeof(float), 16);
    float *y = (float *) _mm_malloc(m*sizeof(float), 16);

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
	//Declaración de los registros necesarios
	__m128 regA[4];
	__m128 regAlfaX;
	__m128 regX;
	__m128 regY;
	__m128 regAdd;
	__m128 regHAdd1, regHAdd2;
	__m128 regAlfa = _mm_set_ps(alfa, alfa, alfa, alfa);

	//Variables necesarias	
	float toRegister[4];
	int c;

    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m; i+=4) {
		regY = _mm_load_ps(&(y[i]));
		regAdd = _mm_setzero_ps();
        for (j=0; (j<n) ; j+=4) {
			regX = _mm_load_ps(&(x[j]));
			//Cargamos os flotantes nos rexistros de 128 bits
			if ( i + 4 > m ) {
				regA[0] = _mm_setzero_ps();
				regA[1] = _mm_setzero_ps();
				regA[2] = _mm_setzero_ps();
				regA[3] = _mm_setzero_ps();
				for ( c=i; c < m; c++ )
					regA[c-i] = fillRegister(c,j,n,A);

			} else if ( j+4 > n){
				memset(toRegister, 0, sizeof(float) * 4);
				for ( c=i; c < i + 4; c++ ){
					regA[c-i] = fillRegister(c,j,n,A);
				}
				
			} else if ( n % 4 != 0){
				memset( toRegister, 0, 4 * sizeof(float) );
				for (c=0; c<4; c++){
				    memcpy (toRegister, &(A[i*n+j+c*n]), 4*sizeof(float));	
					regA[c]=_mm_load_ps(toRegister);
				}

			}else
			{
				regA[0] = _mm_load_ps(&A[i*n+j]);
				regA[1] = _mm_load_ps(&A[i*n+j+n]);
				regA[2] = _mm_load_ps(&A[i*n+j+2*n]);
				regA[3] = _mm_load_ps(&A[i*n+j+3*n]);
			}
			//Multiplicamos os 4 valores almacenados en X por alfa
			regAlfaX = _mm_mul_ps(regX, regAlfa);

			//Multipicamos o anterior, float a float contra os flotantes de A cargados de 4 en 4
			regA[0] = _mm_mul_ps(regAlfaX, regA[0]);
			regA[1] = _mm_mul_ps(regAlfaX, regA[1]);
			regA[2] = _mm_mul_ps(regAlfaX, regA[2]);
			regA[3] = _mm_mul_ps(regAlfaX, regA[3]);

			//Sumanse en horizontal, de dous en dous os elementos de cada rexistro
			//E os resultados almacenanse nun rexistro de 128 bits
			regHAdd1 = _mm_hadd_ps(regA[0], regA[1]);
			regHAdd2 = _mm_hadd_ps(regA[2], regA[3]);

			//O obxetivo e sumar horizontalmente os catro elementos de cada rexistro
			//Dado que co as operacións anterior so sumamos de 2 en 2, necesitamos
			//outra operación de suma horizontal, para que finalmente nos devolva
			//un rexitro que conteña a suma horizontal de todolos elementos
			//de cada rexitro inicial. Por último levamos as contas nun rexistro
			//no que vamos facendo sumas ao resultado anterior para si acabar
			//coa suma das filas.
			regAdd = _mm_add_ps(_mm_hadd_ps(regHAdd1, regHAdd2),regAdd);
		}
		//Unha vez chegado o final de cada conxunto de bloques en horizontal,
		//teremos un rexistro que almacena un máximo de 4 flotantes, que representan
		//O valor de aplicar a fórmula a catro filas, polo que xa podemos almacenalo
		//na variable desexada
		_mm_store_ps(&(y[i]),_mm_add_ps(regAdd,regY));
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

	
    return 0;
}




