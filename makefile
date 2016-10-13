all: p1c autovec p1SSE1 p1SSE2 p1SSE3 p1SSE4

p1c: matrizVectorP1.c
	gcc -O3 -o matrizVectorP1.out matrizVectorP1.c -lm

autovec: matrizVectorP1.c
	gcc -O3 -march=nocona -msse3 -ftree-vectorize -ftree-vectorizer-verbose=2 -o matrizVectorP1Vec.out matrizVectorP1.c -lm

p1SSE1: matrizVectorP1SSE.c
	gcc  -O3 -march=nocona -msse3 -o matrizVectorP1SSE.out matrizVectorP1SSE.c -lm

p1SSE2: matrizVectorP1SSENoHadd.c
	gcc  -O3 -march=nocona -msse3 -o matrizVectorP1SSENoHadd.out matrizVectorP1SSENoHadd.c -lm

p1SSE3: matrizVectorP1SSELoadU.c
	gcc  -O3 -march=nocona -msse3 -o matrizVectorP1SSELoadU.out matrizVectorP1SSELoadU.c -lm

p1SSE4: matrizVectorP1SSENoHaddLoadU.c
	gcc  -O3 -march=nocona -msse3 -o matrizVectorP1SSENoHaddLoadU.out matrizVectorP1SSENoHaddLoadU.c -lm

clean:
	rm -f p3c *.out
