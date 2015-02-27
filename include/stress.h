#ifndef _STRESS_H
#define _STRESS_H

#include "bulk.h"
#include "constants.h"
#include "init.h"
#include "free.h"

void allocateForces(int n1, int n2, int n3);
void deallocateForces(int n1, int n2, int n3);

void allocateStress(int n1, int n2, int n3);
void deallocateStress(int n1, int n2, int n3);

void StressTensor();
void UseForces();

#endif /* _STRESS_H */
