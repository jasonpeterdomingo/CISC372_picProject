
image:image.c image.h
	gcc -g image.c -o image -lm

image_openmp:image_openmp.c image.h
	gcc -fopenmp -g image_openmp.c -o image_openmp -lm

image_pthread: image_pthread.c image.h
	gcc -g image_pthread.c -o image_pthread -lpthread -lm

clean:
	rm -f image image_pthread image_openmp *.o output.png