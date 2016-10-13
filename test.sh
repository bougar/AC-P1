#!/bin/bash
echo "Prubas sin SSE:"
echo "+Sin vectorización autmática:"
./matrizVectorP1.out $1 $2 $3 0
echo "+Con vectorización autmática:"
./matrizVectorP1Vec.out $1 $2 $3 0
echo ""
echo "Pruebas con SSE:"
echo "Usando hadd con loadU:"
./matrizVectorP1SSELoadU.out $1 $2 $3 0
echo "Usando hadd con load:"
./matrizVectorP1SSE.out $1 $2 $3 0
echo "Sin usar hAdd:"
./matrizVectorP1SSENoHadd.out $1 $2 $3 0
echo ""
