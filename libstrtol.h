#ifndef LIBSTRTOL_H_
#define LIBSTRTOL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>

int* int_deserialize(char* string, int *return_row, int *return_col);

#endif /*libstrtol.h*/