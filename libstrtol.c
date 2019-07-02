#include "libstrtol.h"

int* int_deserialize(char* string, int *return_row, int *return_col) {
    int *return_matrix = NULL, *pointer = NULL;
    char *start;
    int row, col;
    int i, j;
    start = string;
    row = (int) strtol(start, &start, 10);
    if (errno == ERANGE){
        printf("range error, got ");
        errno = 0;
        return return_matrix;
    }

    col = (int) strtol(start, &start, 10);
    if (errno == ERANGE){
        printf("range error, got ");
        errno = 0;
        return return_matrix;
    }
    printf("row: %d, col: %d\n", row, col);

    return_matrix = calloc(row * col, sizeof(int *));
    pointer = return_matrix;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            *pointer = (int) strtol(start, &start, 10);
            pointer++;
        }
    }
    *return_row = row;
    *return_col = col;
    return return_matrix;
}