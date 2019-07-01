#include "libstrtol.h"

int** int_deserialize(char* string, int *return_row, int *return_col) {
    int **return_matrix = NULL;
    char *start, *end;
    const char delimiter[2] = "\n";
    int row, col;
    size_t data_len = 16;
    uint64_t read_time = 0, strtol_transform_time = 0;
    int i, j;

    start = strtok(string, delimiter);
    row = (int) strtol(start, &end, 10);
    start = end;
    if (errno == ERANGE){
        printf("range error, got ");
        errno = 0;
        return return_matrix;
    }

    col = (int) strtol(start, &end, 10);
    if (errno == ERANGE){
        printf("range error, got ");
        errno = 0;
        return return_matrix;
    }
    printf("row: %d, col: %d\n", row, col);

    return_matrix = calloc(row, sizeof(int *));
    for (i = 0; i < row; i++) {
        return_matrix[i] = calloc(col, sizeof(int));
    }

    for (i = 0; i < row; i++) {
        start = strtok(NULL, delimiter);

        for (j = 0; j < col; j++) {
            return_matrix[i][j] = (int) strtol(start, &start, 10);
        }
    }
    *return_row = row;
    *return_col = col;
    return return_matrix;
}