#ifndef _FREE_H
#define _FREE_H

#include <stdlib.h>

#include "bulk.h"
#include "constants.h"
#include "stress.h"

void FreeMemory();

void deallocate3D(double***, int, int);

#endif /* _FREE_H */
