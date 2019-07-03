#include <kmcuda.h>
#include "libstrtol.h"

void usage(char** argv) {
    printf("usage: %s <input file>\n", argv[0]);
}

int main(int argc, char** argv) {
    FILE *stream;
    int fd;
    struct stat sb;
    char *start;
    int i, j, row, col;
    int *strtol_matrix = NULL;
    float *samples, *centroids;
    uint32_t *assignments;
    uint64_t total_size = 0, read_time = 0, strtol_transform_time = 0;
    struct timeval start_time, end_time;
    const int clusters_size = 10;
    float average_distance;
    KMCUDAResult result;

    if (argc < 2) {
        usage(argv);
        return 1;
    }

    gettimeofday(&start_time, NULL);
    printf("reading: %s\n", argv[1]);
    fd = open(argv[1], O_RDONLY);
    fstat(fd, &sb);
    start = calloc(sb.st_size, sizeof(char));
    read(fd, start, sb.st_size);
    gettimeofday(&end_time, NULL);
    read_time = ((end_time.tv_sec * 1000000 + end_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec));
    close(fd);

    gettimeofday(&start_time, NULL);
    strtol_matrix = int_deserialize(start, &row, &col);
    gettimeofday(&end_time, NULL);
    strtol_transform_time = ((end_time.tv_sec * 1000000 + end_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec));
    
    if (strtol_matrix == NULL) {
        return 1;
    }

    printf("row: %d, col: %d\n", row, col);
    printf("read time: %lu usec\n", read_time);
    printf("strtol transform time: %lu usec\n", strtol_transform_time);
    printf("total time: %lu usec\n", read_time+strtol_transform_time);
    total_size = ((uint64_t)row) * col;
    samples = calloc(total_size, sizeof(float));
    centroids = calloc(clusters_size * col, sizeof(float));
    assignments = calloc(row, sizeof(float));

    for (i = 0; i < total_size; i++) {
        samples[i] = *strtol_matrix * 1.0f;
    }

    free(strtol_matrix);
    free(start);
    
    result = kmeans_cuda(
        kmcudaInitMethodPlusPlus, NULL,  // kmeans++ centroids initialization
        0.01,                            // less than 1% of the samples are reassigned in the end
        0.1,                             // activate Yinyang refinement with 0.1 threshold
        kmcudaDistanceMetricL2,          // Euclidean distance
        row, col, clusters_size,
        0xDEADBEEF,                      // random generator seed
        0,                               // use all available CUDA devices
        -1,                              // samples are supplied from host
        0,                               // not in float16x2 mode
        1,                               // moderate verbosity
        samples, centroids, assignments, &average_distance);
    
    free(samples);
    free(centroids);
    free(assignments);
    exit(EXIT_SUCCESS);
}
