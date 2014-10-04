#include <stdio.h>
#include "headers/forces.h"
#include "headers/bulk.h"
#include "headers/constants.h"
#include "headers/stress.h"

void InitializeData();
void Input();
void TimeStepSize();
void Phase1();
void Phase2();
void StressTensor();
void UseForces();
void WriteData();

int main (int argc, char** argv)
{
	Input();
	InitializeData();

	NSTEP = 0;
	TIME = 0.0;
	do {
		TimeStepSize();
		NSTEP++;
		TIME += 0.5*DT;
		Phase1();
		StressTensor();
		UseForces();
		Phase2();
		Phase1();
		UseForces();
		TIME += 0.5*DT;

		if (NSTEP % NPRINT == 0) WriteData();
		printf("step: %d dt:%f\n", NSTEP, DT);
	} while (NSTEP < NSTOP);
    return 0;
}

void Input()
{
	NSTOP = 100;
	NPRINT = 10;
}

void InitializeData()
{

}

void TimeStepSize()
{
	DT = 0.1;
}

void Phase1()
{

}

void Phase2()
{

}

void StressTensor()
{

}

void UseForces()
{

}

void WriteData()
{
	//printf("write data\n");
}

