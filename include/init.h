#ifndef _INIT_H
#define _INIT_H

#include <stdlib.h>

#include "bulk.h"
#include "constants.h"
#include "stress.h"
#include "options.h"

void Input(Options opt);
void InitializeData();

double ***allocate3D(int, int, int);

#endif /* _INIT_H */
