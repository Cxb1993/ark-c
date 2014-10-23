#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

	// coordinates of west plane
	X1W = 0.15;
	// coordinates of east plane
	X1E = 0.25;

	// coordinates of south plane
	X2S = 0.0;
	// coordinates of north plane
	X2N = 2.0 * PI;

	// coordinates of bottom plane
	X3B = 0.0;
	// coordinates of top plane
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

	// initial velocity along the X1 axis
	U10=0.;
	// initial velocity along the X2 axis
	U20=0.;
	// initial velocity along the X3 axis
	U30=1.;
	// sound velocity
	SOUND=10.;

	// pressure on the top plane
	POUTLET=0;
	// velocity on the bottom plane along the X3 axis
	U3INLET=U30;
	// velocity on the bottom plane along the X2 axis
	U2INLET=U20;
	// velocity on the bottom plane along the X1 axis
	U1INLET=U10;
	// temperature on the bottom plane
	TINLET=T0;
	// unperturbed density of the liquid
	RO0G=1.;
	// unperturbed density of the borders material
	RO0S=100000000.;

	// #####################################################
	// 				block of arrays allocation
	// #####################################################

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
//	double	ALFA0 = 0.204,
//			BETA = 0.3,
//			R0 = 0.05;

	// init value of time step
	DT = pow(10, -4);

	// grid step along the X1 axis
	DX1=(X1E-X1W)/(N1-1);
	DX2=(X2N-X2S)/(N2-1);
	DX3=(X3T-X3B)/(N3-1);

	X1[0] = X1W - DX1;
	X2[0] = X2S - DX2;
	X3[0] = X3B - DX3;

	// block of arrays initialization
	for (int i = 0; i <= N1 ; ++i) {
		X1[i+1] = X1[i] + DX1;
	}

	for (int j = 0; j <= N2 ; ++j) {
		X2[j +1] = X2[j] + DX2;
	}

	for (int k = 0; k <= N3 ; ++k) {
		X3[k +1] = X3[k] + DX3;
	}

	for (int i = 0; i <= N1; ++i) {
		for (int j = 0; j <= N2; ++j) {
			for (int k = 0; k <= N3; ++k) {
				U1CON[i][j][k] = U1NCON[i][j][k] = U10;
				U2CON[i][j][k] = U2NCON[i][j][k] = U20;
				U3CON[i][j][k] = U3NCON[i][j][k] = U30;
				ROCON[i][j][k] = RONCON[i][j][k] = RO0G;
				TCON[i][j][k] = TNCON[i][j][k] = T0;

				P1[i][j][k] = P2[i][j][k] = P3[i][j][k] = 0.;
				RO1[i][j][k] = RO2[i][j][k] = RO3[i][j][k] = RO0G;
				U11[i][j][k] = U12[i][j][k] = U13[i][j][k] = U10;
				U21[i][j][k] = U22[i][j][k] = U23[i][j][k] = U20;
				U31[i][j][k] = U32[i][j][k] = U33[i][j][k] = U30;
				T1[i][j][k] = T2[i][j][k] = T3[i][j][k] = T0;
			}

		}
	}

	for (int j = 0; j <= N2; ++j) {
		for (int k = 0; k <= N3; ++k) {
			P1[N1+1][j][k] = 0.;
			RO1[N1+1][j][k] = RO0G;
			U11[N1+1][j][k] = U10;
			U21[N1+1][j][k] = U20;
			U31[N1+1][j][k] = U30;
			T1[N1+1][j][k] = T0;
		}
	}

	for (int i = 0; i <= N1; ++i) {
		for (int k = 0; k <= N3; ++k) {
			P2[i][N2+1][k] = 0.;
			RO2[i][N2+1][k] = RO0G;
			U12[i][N2+1][k] = U10;
			U22[i][N2+1][k] = U20;
			U32[i][N2+1][k] = U30;
			T2[i][N2+1][k] = T0;
		}
	}

	for (int i = 0; i <= N1; ++i) {
		for (int j = 0; j <= N2; ++j) {
			P3[i][j][N3+1] = 0.;
			RO3[i][j][N3+1] = RO0G;
			U13[i][j][N3+1] = U10;
			U23[i][j][N3+1] = U20;
			U33[i][j][N3+1] = U30;
			T3[i][j][N3+1] = T0;
		}
	}

}

void TimeStepSize()
{
	double X1C, X1L, U1C, U2C, U3C, DTU1, DTU2, DTU3, DTU, DTV1, DTV2, DTV3, DTV;

	for (int i = 1; i < N1; ++i) {
		for (int j = 1; j < N2; ++j) {
			for (int k = 1; k < N3; ++k) {
				X1C = (X1[i+1] + X1[i])/2;
				X1L = 1+(L-1)*(X1C-1);

				U1C = U1CON[i][j][k];
				U2C = U2CON[i][j][k];
				U3C = U3CON[i][j][k];

				DTU1 = CFL*DX1/(SOUND + fabs(U1C));
				DTU2 = CFL*X1L*DX2/(SOUND + fabs(U2C));
				DTU3 = CFL*DX3/(SOUND + fabs(U3C));

				// DTU = MIN(DTU1, DTU2, DTU3)
				DTU = DTU1 > DTU2 ? DTU2 : DTU1;
				DTU = DTU > DTU3 ? DTU3 : DTU;

				if (VIS > pow(10, -16)) {
					DTV1 = CFL*DX1*DX1/(2.*VIS);
					DTV2 = CFL*(X1L*DX2)*(X1L*DX2)/(2.*VIS);
					DTV3 = CFL*DX3*DX3/(2.*VIS);

					// DTV = MIN (DTV1, DTV2, DTV3)
					DTV = DTV1 > DTV2 ? DTV2 : DTV1;
					DTV = DTV > DTV3 ? DTV3 : DTV;
				} else {
					DTV = pow(10, 16);
				}
				double dttemp = DTU > DTV ? DTV : DTU;
				if (dttemp < DT) DT = dttemp;
			}
		}
	}
}

void Phase1()
{
	// 
	for (int i = 1; i < N1; i++)
	{
		for (int j = 1; j < N2; j++)
		{
			for (int k = 1; k < N3; k++)
			{
				U1CON[i][j][k] = U1NCON[i][j][k];
				U1CON[i][j][k] = U1NCON[i][j][k];
				U1CON[i][j][k] = U1NCON[i][j][k];
				ROCON[i][j][k] = RONCON[i][j][k];
				TCON[i][j][k] = TNCON[i][j][k];
			}
		}
	}

	double DS1, DS2, DS3, DVC, XLE, XLW, XLT, XLB, ALFA;

			// velocity, density, temperature and pressure on the eastern plane
	double	U1E, U2E, U3E, ROE, TE, PE,
			// velocity, density , temperature and pressure on the western plane
			U1W, U2W, U3W, ROW, TW, PW,
			// velocity, density , temperature and pressure on the northern plane
			U1N, U2N, U3N, RON, TN, PN,
			// velocity, density , temperature and pressure on the southern plane
			U1S, U2S, U3S, ROS, TS, PS,
			// velocity, density , temperature and pressure on the top plane
			U1T, U2T, U3T, ROT, TT, PT,
			// velocity, density , temperature and pressure on the bottom plane
			U1B, U2B, U3B, ROB, TB, PB,
			// velocity, density and temperature in the cell center
			U1C, U2C, U3C, ROC, TC,
			// velocity, density and temperature in the cell center on the next time step
			U1CN, U2CN, U3CN, ROCN, TCN;

	// plane squares
	DS1 = DX2*DX3;
	DS2 = DX1*DX3; 
	DS3 = DX1*DX2;

	for (int i = 1; i < N1; i++)
	{
		for (int j = 1; j < N2; j++)
		{
			for (int k = 1; k < N3; k++)
			{
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				XLE = 1 + (L - 1)*(X1[i] - 1);
				// geometric factor of cylindricity
				XLW = 1 + (L - 1)*(X1[i - 1] - 1);
				// geometric factor of cylindricity
				XLT = 0.5*(XLE + XLW);
				// geometric factor of cylindricity
				XLB = XLT;
				// cell volume
				DVC = 0.5*(XLE + XLW)*DX1*DX2*DX3;
				// geometric parameter
				ALFA = DT*(L - 1)*DX1*DX2*DX3 / (2 * DVC);

				// #########################################################
				//	get velocity, density , temperature and pressure values
				// #########################################################

				// east plane
				U1E = U11[i + 1][j][k];
				U2E = U21[i + 1][j][k];
				U3E = U31[i + 1][j][k];
				ROE = RO1[i + 1][j][k];
				TE = T1[i + 1][j][k];
				PE = P1[i + 1][j][k];
				// west plane
				U1W = U11[i][j][k];
				U2W = U21[i][j][k];
				U3W = U31[i][j][k];
				ROW = RO1[i][j][k];
				TW = T1[i][j][k];
				PW = P1[i][j][k];

				// north plane
				U1N = U12[i][j + 1][k];
				U2N = U22[i][j + 1][k];
				U3N = U32[i][j + 1][k];
				RON = RO2[i][j + 1][k];
				TN = T2[i][j + 1][k];
				PN = P2[i][j + 1][k];
				// south plane
				U1W = U12[i][j][k];
				U2W = U22[i][j][k];
				U3W = U32[i][j][k];
				ROW = RO2[i][j][k];
				TS = T2[i][j][k];
				PS = P2[i][j][k];

				// top plane
				U1T = U13[i][j][k + 1];
				U2T = U23[i][j][k + 1];
				U3T = U33[i][j][k + 1];
				ROT = RO3[i][j][k + 1];
				TT = T3[i][j][k + 1];
				PT = P3[i][j][k + 1];
				// bottom plane
				U1B = U13[i][j][k];
				U2B = U23[i][j][k];
				U3B = U33[i][j][k];
				ROB = RO3[i][j][k];
				TB = T3[i][j][k];
				PB = P3[i][j][k];

				// cell center
				U1C = U1CON[i][j][k];
				U2C = U2CON[i][j][k];
				U3C = U3CON[i][j][k];
				ROC = ROCON[i][j][k];
				TC = TCON[i][j][k];
				
				// #####################################################
				//					new values evaluating
				// #####################################################

				// new density
				ROCN = (ROC*DVC - 0.5*DT*((XLE*ROE*U1E - XLW*ROW*U1W)*DS1 +
					(RON*U2N - ROS*U2S)*DS2 + (XLT*ROT*U3T - XLB*ROB*U3B)*DS3)) / DVC;
				
				// new conservative velocity along the X1 axis
				double U1CP = (ROC*U1C*DVC - 0.5*DT*((RON*U2N*U1N - ROS*U2S*U1S)*DS2 +
					(XLT*ROT*U3T*U1T - XLB*ROB*U3B*U1B)*DS3 +
					(XLE*(ROE*U1E*ROE*U1E + PE) - XLW*(ROW*U1W*ROW*U1W + PW))*DS1
					- 0.5*(L - 1)*(PE + PW)*DX1*DX2*DX3)) / (DVC*ROCN);
				// new conservative velocity along the X2 axis
				double U2CP = U2CP = (ROC*U2C*DVC - 0.5*DT*((XLE*ROE*U1E*U2E - XLW*ROW*U1W*U2W)*DS1 +
					((RON*U2N*RON*U2N + PN) - (ROS*U2S*ROS*U2S + PS))*DS2 +
					(XLT*ROT*U3T*U2T - XLB*ROB*U3B*U2B)*DS3)) / (DVC*ROCN);

				// take into account of centrifugal and Coriolis forces
				U1CN = (U1CP - ALFA*U2C*U1CP) / (1 + (ALFA*U2C)*(ALFA*U2C));
				U2CN = U2CP - ALFA*U2C*U1CN;

				// new conservative velocity along the X3 axis
				U3CN = (ROC*U3C*DVC - 0.5*DT*((XLE*ROE*U1E*U3E - XLW*ROW*U1W*U3W)*DS1 +
					(RON*U2N*U3N - ROS*U2S*U3S)*DS2 +
					(XLT*(ROT*U3T*ROT*U3T + PT) - XLB*(ROB*U3B*ROB*U3B + PB))*DS3)) / (DVC*ROCN);

				// new temperature
				TCN = (ROC*TC*DVC - 0.5*DT*((XLE*ROE*TE*U1E - XLW*ROW*TW*U1W)*DS1 +
					(RON*TN*U2N - ROS*TS*U2S)*DS2 +
					(XLT*ROT*TT*U3T - XLB*ROB*TB*U3B)*DS3)) / (DVC*ROCN);

				// finally
				U1NCON[i][j][k] = U1CN;
				U2NCON[i][j][k] = U2CN;
				U3NCON[i][j][k] = U3CN;
				RONCON[i][j][k] = ROCN;
				TNCON[i][j][k] = TCN;
			}
		}
	}

	// periodicity conditions
	for (int i = 1; i <= N1; ++i)
	{
		for (int k = 1; k <= N3; ++k)
		{
			// periodicity condition on the north plane
			U1NCON[i][N2][k] = U1NCON[i][1][k];
			U2NCON[i][N2][k] = U2NCON[i][1][k];
			U3NCON[i][N2][k] = U3NCON[i][1][k];
			RONCON[i][N2][k] = RONCON[i][1][k];
			TNCON[i][N2][k] = TNCON[i][1][k];
			// periodicity condition on the south plane
			U1NCON[i][0][k] = U1NCON[i][N2 - 1][k];
			U2NCON[i][0][k] = U2NCON[i][N2 - 1][k];
			U3NCON[i][0][k] = U3NCON[i][N2 - 1][k];
			RONCON[i][0][k] = RONCON[i][N2 - 1][k];
			TNCON[i][0][k] = TNCON[i][N2 - 1][k];
		}
	}	

	// no-slip conditions
	for (int j = 1; j < N2; ++j)
	{
		for (int k = 1; k < N3; ++k)
		{
			// no-slip consition on the west plane
			U1NCON[0][j][k] = 0.;
			U2NCON[0][j][k] = 0.;
			U3NCON[0][j][k] = 0.;
			RONCON[0][j][k] = RONCON[1][j][k];
			TNCON[0][j][k] = T0;

			// no-slip consition on the east plane
			U1NCON[N1][j][k] = 0.;
			U2NCON[N1][j][k] = 0.;
			U3NCON[N1][j][k] = 0.;
			RONCON[N1][j][k] = RONCON[N1 - 1][j][k];
			TNCON[N1][j][k] = T0;
		}
	}
}

void Phase2()
{

}

void StressTensor()
{
	// initialization of friction stress arrays
	for (int i = 0; i <= N1; ++i) {
		for (int j = 0; j <= N2; ++j) {
			for (int k = 0; k <= N3; ++k) {
				SIGM11[i][j][k] = 0.0;
				SIGM21[i][j][k] = 0.0;
				SIGM31[i][j][k] = 0.0;

				SIGM12[i][j][k] = 0.0;
				SIGM22[i][j][k] = 0.0;
				SIGM32[i][j][k] = 0.0;

				SIGM13[i][j][k] = 0.0;
				SIGM23[i][j][k] = 0.0;
				SIGM33[i][j][k] = 0.0;
			}
		}
	}

	// #####################################################
	// 				boundary conditions
	// #####################################################

	// no-slip condition on the boundary faces perpendicular to X1
	X1[0] = X1[1];
	X1[N1+1] = X1[N1];

	for (int j = 1; j < N2 ; ++j) {
		for (int k = 1; k < N3; ++k) {
			U1CON[0][j][k] = 0.;
			U2CON[0][j][k] = 0.;
			U3CON[0][j][k] = 0.;

			U1CON[N1][j][k] = 0.;
			U2CON[N1][j][k] = 0.;
			U3CON[N1][j][k] = 0.;
		}

	}

	// periodic contition on the boundary faces perpendicular to X2
	// nothing to do since we have uniform grid

	// periodic contition on the boundary faces perpendicular to X3
	// nothing to do since we have uniform grid

	// #####################################################
	// 				crawling along the faces
	// #####################################################

	double XLE, XLW, XLT, XLN, U1C, U1CE, U2C, U2CE, U3C, U3CE;

	// crawling along the face perpendicular to X1
	for (int i = 1; i <= N1; ++i) {
		for (int j = 1; j < N2; ++j) {
			for (int k = 1; k < N3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				XLE = 1 + (L-1)*(X1[i] - 1);

				// velocity components in cell centers
				U1C = U1CON[i][j][k];
				U1CE = U1CON[i-1][j][k];

				U2C = U2CON[i][j][k];
				U2CE = U2CON[i][j][k];

				U3C = U3CON[i][j][k];
				U3CE = U3CON[i-1][j][k];

				// friction stress
				SIGM11[i][j][k]=-VIS*XLE*(U1CE-U1C)/DX1;
				SIGM21[i][j][k]=-VIS*XLE*(U2CE-U2C)/DX1;
				SIGM31[i][j][k]=-VIS*XLE*(U3CE-U3C)/DX1;
			}
		}
	}

	double U1CN, U2CN, U3CN;

	// crawling along the face perpenditcular to X2
	for (int i = 1; i < N1; ++i) {
		for (int j = 1; j <= N2; ++j) {
			for (int k = 1; k < N3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				XLE = 1 + (L-1)*(X1[i] - 1);
				// geometric factor of cylindricity
				XLW = 1 + (L-1)*(X1[i-1] - 1);
				// geometric factor of cylindricity
				XLT = 0.5*(XLE + XLW);
				// geometric factor of cylindricity
				XLN = XLT;

				// velocity components in cell centers
				U1C = U1CON[i][j][k];
				U1CN = U1CON[i][j-1][k];

				U2C = U2CON[i][j][k];
				U2CN = U2CON[i][j-1][k];

				U3C = U3CON[i][j][k];
				U3CN = U3CON[i][j-1][k];

				// friction stress
				SIGM12[i][j][k]=-VIS*((U1CN-U1C)/DX2 -(L-1)*(U2C+U2CN))/XLN;
				SIGM22[i][j][k]=-VIS*((U2CN-U2C)/DX2 +(L-1)*(U1C+U1CN))/XLN;
				SIGM32[i][j][k]=-VIS*(U3CN-U3C)/DX2;
			}
		}
	}

	double U1CT, U2CT, U3CT;

	// crawling along the face perpenditcular to X3
	for (int i = 1; i < N1; ++i) {
		for (int j = 1; j < N2; ++j) {
			for (int k = 1; k <= N3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				XLE = 1 + (L-1)*(X1[i] - 1);
				// geometric factor of cylindricity
				XLW = 1 + (L-1)*(X1[i-1] - 1);
				// geometric factor of cylindricity
				XLT = 0.5*(XLE + XLW);

				// velocity components in the cell centers
				U1C = U1CON[i][j][k];
				U1CT = U1CON[i][j][k-1];

				U2C = U2CON[i][j][k];
				U2CT = U2CON[i][j][k-1];

				U3C = U3CON[i][j][k];
				U3CT = U3CON[i][j][k-1];

				// friction stress
				SIGM13[i][j][k]=-VIS*XLT*(U1CT-U1C)/DX3;
				SIGM23[i][j][k]=-VIS*XLT*(U2CT-U2C)/DX3;
				SIGM33[i][j][k]=-VIS*XLT*(U3CT-U3C)/DX3;
			}
		}
	}

	// #####################################################
	// 				friction forces computation
	// #####################################################

	double DS1, DS2, DS3, SIGM1C, SIGM2C;

	// area of the face perpendicuar to X1
	DS1 = DX2*DX3;
	DS2 = DX1*DX3;
	DS3 = DX1*DX2;

	for (int i = 1; i < N1; ++i) {
		for (int j = 1; j < N2; ++j) {
			for (int k = 1; k < N3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				XLE = 1 + (L-1)*(X1[i] - 1);
				// geometric factor of cylindricity
				XLW = 1 + (L-1)*(X1[i-1] - 1);
				// geometric factor of cylindricity
				XLT = 0.5*(XLE + XLW);

				SIGM1C = VIS*U1CON[i][j][k]/XLT;
				SIGM2C = VIS*U2CON[i][j][k]/XLT;

				// friction forces
				F1[i][j][k] =
						(SIGM11[i+1][j][k] - SIGM11[i][j][k]) * DS1 +
						(SIGM12[i][j+1][k] - SIGM12[i][j][k]) * DS2 +
						(SIGM13[i][j][k+1] - SIGM13[i][j][k]) * DS3 -
						(L-1)*SIGM1C*DX1*DX2*DX3;

				F2[i][j][k] =
						(SIGM21[i+1][j][k] - SIGM21[i][j][k]) * DS1 +
						(SIGM22[i][j+1][k] - SIGM22[i][j][k]) * DS2 +
						(SIGM23[i][j][k+1] - SIGM23[i][j][k]) * DS3 -
						(L-1)*SIGM2C*DX1*DX2*DX3;

				F3[i][j][k] =
						(SIGM21[i+1][j][k] - SIGM21[i][j][k]) * DS1 +
						(SIGM22[i][j+1][k] - SIGM22[i][j][k]) * DS2 +
						(SIGM23[i][j][k+1] - SIGM23[i][j][k]) * DS3;
			}
		}
	}

}

void UseForces()
{
	double XLE, XLW, DVC, ROC, ROCN;

	for (int i = 1; i < N1; ++i) {
		for (int j = 1; j < N2; ++j) {
			for (int k = 1; k < N3; ++k) {
				// geometric factor of cylindricity
				XLE = 1+(L-1)*(X1[i+1]-1);
				XLW = 1+(L-1)*(X1[i]-1);
				// cell volume
				DVC = 0.5*(XLE+XLW)*DX1*DX2*DX3;

				ROC = ROCON[i][j][k];
				ROCN = RONCON[i][j][k];

				U1NCON[i][j][k] = (ROC*DVC*U1NCON[i][j][k] + 0.5*DT*F1[i][j][k])/(DVC*ROCN);
				U2NCON[i][j][k] = (ROC*DVC*U2NCON[i][j][k] + 0.5*DT*F2[i][j][k])/(DVC*ROCN);
				U3NCON[i][j][k] = (ROC*DVC*U3NCON[i][j][k] + 0.5*DT*F3[i][j][k])/(DVC*ROCN);
			}
		}
	}

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
		free(arr[i]);
	}
	free(arr);
	arr = NULL;
}