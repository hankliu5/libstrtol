#include "libstrtol.h"

void usage(char** argv) {
    printf("usage: %s <input file>\n", argv[0]);
}

int main(int argc, char** argv) {
    FILE *stream;
    int fd;
    struct stat sb;
    char *start, *end;
    int row, col, i, j;
    size_t data_len = 16;
    int *strtol_matrix = NULL;
    uint64_t read_time = 0, strtol_transform_time = 0;
    struct timeval start_time, end_time;

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
    
    free(strtol_matrix);
    free(start);
    close(fd);
    exit(EXIT_SUCCESS);
}
