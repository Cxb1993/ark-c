#include <stdio.h>
#include <stdlib.h>
#include "headers/forces.h"
#include "headers/bulk.h"
#include "headers/constants.h"
#include "headers/stress.h"

void Input();
void InitializeData();
void TimeStepSize();
void Phase1();
void Phase2();
void StressTensor();
void UseForces();
void WriteData();
void FreeMemory();

double ***allocate3D(int, int, int);
void deallocate3D(double***, int, int);

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

	FreeMemory();
    return 0;
}

void Input()
{
	// geometry index
	L = 1;

	// total number of grid nodes along the X1 axis
	N1G = 32;
	// total number of grid nodes along the X2 axis
	N2G = 32;
	// total number of grid nodes along the X3 axis
	N3G = 64;

	// number of grid nodes along the X1 axis to 1 processor
	N1 = N1G/1;
	// number of grid nodes along the X2 axis to 1 processor
	N2 = N2G/1;
	// number of grid nodes along the X3 axis to 1 processor
	N3 = N3G/1;

	// coordinates of west border
	X1W = 0.15;
	// coordinates of east border
	X1E = 0.25;

	// coordinates of south border
	X2S = 0.0;
	// coordinates of north border
	X2N = 2.0 * PI;

	// coordinates of bottom border
	X3B = 0.0;
	// coordinates of top border
	X3T = 1.0;

	// total number of steps
	NSTOP = 100;
	// print interval
	NPRINT = 100;

	// Courant number
	CFL = 0.2;

	// kinematic viscosity
	VIS=0.5/50.;  // 0.5
	// initial temperature
	T0=1.;

	// initial speed along the X1 axis
	U10=0.;
	// initial speed along the X2 axis
	U20=0.;
	// initial speed along the X3 axis
	U30=1.;
	// sound speed
	SOUND=10.;

	// pressure on the top border
	POUTLET=0;
	// speed on the bottom border along the X3 axis
	U3INLET=U30;
	// speed on the bottom border along the X2 axis
	U2INLET=U20;
	// speed on the bottom border along the X1 axis
	U1INLET=U10;
	// temperature on the bottom border
	TINLET=T0;
	// unperturbed density of the liquid
	RO0G=1.;
	// unperturbed density of the borders material
	RO0S=100000000.;

	// block of arrays allocation
	// coordinates of grid nodes along the all of the axises
	X1 = calloc((size_t) (N1+2), sizeof(double));
	X2 = calloc((size_t) (N2+2), sizeof(double));
	X3 = calloc((size_t) (N3+2), sizeof(double));

	// variables on the current time step
	ROCON	= allocate3D(N1, N2, N3);
	TCON	= allocate3D(N1, N2, N3);
	U1CON	= allocate3D(N1, N2, N3);
	U2CON	= allocate3D(N1, N2, N3);
	U3CON	= allocate3D(N1, N2, N3);

	// variables on the next time step
	RONCON	= allocate3D(N1, N3, N3);
	TNCON	= allocate3D(N1, N2, N3);
	U1NCON	= allocate3D(N1, N2, N3);
	U2NCON	= allocate3D(N1, N2, N3);
	U3NCON	= allocate3D(N1, N2, N3);

	// variables perpendicular to the axis X1
	RO1	= allocate3D(N1+1, N2, N3);
	T1	= allocate3D(N1+1, N2, N3);
	U11	= allocate3D(N1+1, N2, N3);
	U21	= allocate3D(N1+1, N2, N3);
	U31	= allocate3D(N1+1, N2, N3);
	P1	= allocate3D(N1+1, N2, N3);

	// variables perpendicular to the axis X2
	RO2	= allocate3D(N1, N2+1, N3);
	T2	= allocate3D(N1, N2+1, N3);
	U12	= allocate3D(N1, N2+1, N3);
	U22	= allocate3D(N1, N2+1, N3);
	U32	= allocate3D(N1, N2+1, N3);
	P2	= allocate3D(N1, N2+1, N3);

	// variables perpendicular to the axis X3
	RO3	= allocate3D(N1, N2, N3+1);
	T3	= allocate3D(N1, N2, N3+1);
	U13	= allocate3D(N1, N2, N3+1);
	U23	= allocate3D(N1, N2, N3+1);
	U33	= allocate3D(N1, N2, N3+1);
	P3	= allocate3D(N1, N2, N3+1);

	// forces
	F1 = allocate3D(N1, N2, N3);
	F2 = allocate3D(N1, N2, N3);
	F3 = allocate3D(N1, N2, N3);

	// get into NMAX maxim of N1, N2, N3
	int NMAX = (N1 > N2 ? N1 : N2);
	NMAX = NMAX > N3 ? NMAX : N3;

	// additional buffers for phase 2
	RBUF	= calloc((size_t) (NMAX+2), sizeof(double));
	QBUF	= calloc((size_t) (NMAX+1), sizeof(double));
	TFBUF	= calloc((size_t) (NMAX+2), sizeof(double));
	TBBUF	= calloc((size_t) (NMAX+2), sizeof(double));
	U2FBUF	= calloc((size_t) (NMAX+2), sizeof(double));
	U2BBUF	= calloc((size_t) (NMAX+2), sizeof(double));
	U3FBUF	= calloc((size_t) (NMAX+2), sizeof(double));
	U3BBUF	= calloc((size_t) (NMAX+2), sizeof(double));

	// friction stress
	SIGM11 = allocate3D(N1, N2, N3);
	SIGM21 = allocate3D(N1, N2, N3);
	SIGM31 = allocate3D(N1, N2, N3);

	SIGM12 = allocate3D(N1, N2, N3);
	SIGM22 = allocate3D(N1, N2, N3);
	SIGM32 = allocate3D(N1, N2, N3);

	SIGM13 = allocate3D(N1, N2, N3);
	SIGM23 = allocate3D(N1, N2, N3);
	SIGM33 = allocate3D(N1, N2, N3);
}

void InitializeData()
{
	// grid step along the X1 axis
	DX1=(X1E-X1W)/(N1-1);
	DX2=(X2N-X2S)/(N2-1);
	DX3=(X3T-X3B)/(N3-1);

	X1[0] = X1W - DX1;
	X2[0] = X2S - DX2;
	X3[0] = X3B - DX3;

	//
	for (int i = 0; i <= N1 ; ++i) {
		X1[i+1] = X1[i] + DX1;
	}

	for (int j = 0; j <= N2 ; ++j) {
		X2[j +1] = X2[j] + DX2;
	}

	for (int k = 0; k <= N3 ; ++k) {
		X3[k +1] = X3[k] + DX3;
	}
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

void FreeMemory()
{
	free(X1);
	free(X2);
	free(X3);

	deallocate3D(ROCON, N1, N2);
	deallocate3D(U1CON, N1, N2);
	deallocate3D(U2CON, N1, N2);
	deallocate3D(U3CON, N1, N2);
	deallocate3D(TCON, N1, N2);

	deallocate3D(RONCON, N1, N2);
	deallocate3D(U1NCON, N1, N2);
	deallocate3D(U2NCON, N1, N2);
	deallocate3D(U3NCON, N1, N2);
	deallocate3D(TNCON, N1, N2);

	deallocate3D(RO1, N1+1, N2);
	deallocate3D(T1, N1+1, N2);
	deallocate3D(U11, N1+1, N2);
	deallocate3D(U21, N1+1, N2);
	deallocate3D(U31, N1+1, N2);
	deallocate3D(P1, N1+1, N2);

	deallocate3D(RO2, N1, N2+1);
	deallocate3D(T2, N1, N2+1);
	deallocate3D(U12, N1, N2+1);
	deallocate3D(U22, N1, N2+1);
	deallocate3D(U32, N1, N2+1);
	deallocate3D(P2, N1, N2+1);

	deallocate3D(RO1, N1, N2);
	deallocate3D(T1, N1, N2);
	deallocate3D(U11, N1, N2);
	deallocate3D(U21, N1, N2);
	deallocate3D(U31, N1, N2);
	deallocate3D(P1, N1, N2);

	deallocate3D(F1, N1, N2);
	deallocate3D(F2, N1, N2);
	deallocate3D(F3, N1, N2);

	free(RBUF);
	free(QBUF);
	free(TFBUF);
	free(TBBUF);
	free(U2FBUF);
	free(U2BBUF);
	free(U3FBUF);
	free(U3BBUF);

	deallocate3D(SIGM11, N1, N2);
	deallocate3D(SIGM21, N1, N2);
	deallocate3D(SIGM31, N1, N2);

	deallocate3D(SIGM12, N1, N2);
	deallocate3D(SIGM22, N1, N2);
	deallocate3D(SIGM32, N1, N2);

	deallocate3D(SIGM13, N1, N2);
	deallocate3D(SIGM23, N1, N2);
	deallocate3D(SIGM33, N1, N2);
}

double ***allocate3D(int n1, int n2, int n3)
{
	double ***arr;

	arr = calloc((size_t)(n1+1), sizeof(double**));

	for (int i = 0; i <= n1; ++i) {
		arr[i] = calloc((size_t)(n2+1), sizeof(double*));
		for (int j = 0; j <= n2; ++j) {
			arr[i][j] = calloc((size_t)(n3+1), sizeof(double));
		}
	}

	return arr;
}

void deallocate3D(double*** arr, int n1, int n2)
{
	for (int i = 0; i <= n1; ++i) {
		for (int j = 0; j <= n2; ++j) {
			free(arr[i][j]);
		}
	}
}