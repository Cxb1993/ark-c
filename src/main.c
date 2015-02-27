#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bulk.h"
#include "constants.h"
#include "init.h"
#include "phases.h"
#include "stress.h"
#include "free.h"
#include "options.h"

void TimeStepSize();
void WriteData();
void WriteDataParaView();

int main (int argc, char** argv)
{
	Options opt = parseOptions(argc, argv);
	infoPrint(opt);
	if (opt.mode < 0) {
		errorPrint(opt);
		return -1;
	} else if (opt.mode == 1) {
		helpPrint(opt);
		return 0;
	} else if (opt.mode == 2) {
		versionPrint(opt);
		return 0;
	} else if (opt.mode > 0) {
		errorPrint(opt);
		return -1;
	}


	Input(opt);
	InitializeData();
	return 0;

	nStep = 0;
	TIME = 0.0;
	do
	{
		TimeStepSize();
		nStep++;
		TIME += 0.5*dt;
		Phase1();
		StressTensor();
		UseForces();
		Phase2();
		Phase1();
		StressTensor();
		UseForces();
		TIME += 0.5*dt;

		if (nStep % nPrint == 0) WriteDataParaView();
		printf("step: %d dt:%E\n", nStep, dt);
	} while (nStep < nStop);
	FreeMemory();
    return 0;
}

// input: l, n1, n2, n3, x1, u1Con, u2Con, u3Con, CFL, dx1, dx2, dx3, VIS;
// output: dt;
void TimeStepSize()
{
	double x1c, x1l, u1_c, u2_c, u3_c, dtu1, dtu2, dtu3, dtu, dtv1, dtv2, dtv3, dtv;

	dt = pow(10, 8);

	for (int i = 1; i < n2; ++i) {
		for (int j = 1; j < n1; ++j) {
			for (int k = 1; k < n3; ++k) {
				x1c = (x1[j+1] + x1[j])/2;
				x1l = 1+(l-1)*(x1c-1);

				u1_c = u1Con[i][j][k];
				u2_c = u2Con[i][j][k];
				u3_c = u3Con[i][j][k];

				dtu1 = CFL*dx1/(sound + fabs(u1_c));
				dtu2 = CFL*x1l*dx2/(sound + fabs(u2_c));
				dtu3 = CFL*dx3/(sound + fabs(u3_c));

				// DTU = MIN(DTU1, DTU2, DTU3)
				dtu = dtu1 > dtu2 ? dtu2 : dtu1;
				dtu = dtu > dtu3 ? dtu3 : dtu;

				if (VIS > pow(10, -16)) {
					dtv1 = CFL*dx1*dx1/(2.*VIS);
					dtv2 = CFL*(x1l*dx2)*(x1l*dx2)/(2.*VIS);
					dtv3 = CFL*dx3*dx3/(2.*VIS);

					// DTV = MIN (DTV1, DTV2, DTV3)
					dtv = dtv1 > dtv2 ? dtv2 : dtv1;
					dtv = dtv > dtv3 ? dtv3 : dtv;
				} else {
					dtv = pow(10, 16);
				}

				// DT = MIN(DT, DTU, DTV)
				dt = dt > dtu ? dtu : dt;
				dt = dt > dtv ? dtv : dt;
			}
		}
	}
}

// input:  nStep, filename, TIME, n1, n2, n3, x1, x2, x3, u1Con, u2Con, u3Con, ronCon, tnCon
// output: nothing
// ??? tnCon or tCon, ronCon or roCon
void WriteData()
{
	char filename[50];

	sprintf(filename, "out_%d.tec", nStep);

	FILE *fd = fopen(filename, "w");

	fprintf(fd, "TITLE=\"OUT\"\nVARIABLES=\"X\",\"Y\",\"Z\",\"U1\",\"U2\",\"U3\",\"PC\",\"TC\"\nZONE T=\"%10.4f\", I=%4d, J=%4d, K=%4d, DATAPACKING=BLOCK\n\nVARLOCATION=([4-8]=CELLCENTERED)\n", TIME, n1, n2, n3);

	for (int i = 0; i < n1; ++i)
	{
		for (int j = 0; j < n2; ++j)
		{
			for (int k = 0; k < n3; ++k)
			{
				fprintf(fd, "%16.8E %16.8E %16.8E %16.8E %16.8E %16.8E %16.8E %16.8E\n", x1[i], x2[j], x3[k], u1Con[i][j][k], u2Con[i][j][k], u3Con[i][j][k], ronCon[i][j][k], tnCon[i][j][k]);
			}
		}
	}

	fclose(fd);
}

// input:  nStep, filename, TIME, n1, n2, n3, x1, x2, x3, u1nCon, u2nCon, u3nCon, ronCon, tnCon
// output: nothing
void WriteDataParaView()
{
	char filename[50];

	sprintf(filename, "out_%d.vtk", nStep);

	FILE *fd = fopen(filename, "w");

	fprintf(fd, "# vtk DataFile Version 3.0\nvtk output\nASCII\n");
	fprintf(fd, "DATASET RECTILINEAR_GRID\nDIMENSIONS %d %d %d", n1, n2, n3);

	fprintf(fd, "\nX_COORDINATES %d float\n", n1);
	for (int i = 1; i <= n1; i++)
	{
		fprintf(fd, "%f ", x1[i]);
	}

	fprintf(fd, "\nY_COORDINATES %d float\n", n2);
	for (int i = 1; i <= n2; i++)
	{
		fprintf(fd, "%f ", x2[i]);
	}

	fprintf(fd, "\nZ_COORDINATES %d float\n", n3);
	for (int i = 1; i <= n3; i++)
	{
		fprintf(fd, "%f ", x3[i]);
	}

	fprintf(fd, "\nCELL_DATA %d\nscalars U1 float\nLOOKUP_TABLE default\n", (n1-1)*(n2-1)*(n3-1));
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n2; i++)
		{
			for (int j = 1; j < n1; j++)
			{
				fprintf(fd, "%f ", u1nCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars U2 float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n2; i++)
		{
			for (int j = 1; j < n1; j++)
			{
				fprintf(fd, "%f ", u2nCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars U3 float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n2; i++)
		{
			for (int j = 1; j < n1; j++)
			{
				fprintf(fd, "%f ", u3nCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars PC float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n2; i++)
		{
			for (int j = 1; j < n1; j++)
			{
				fprintf(fd, "%f ", ronCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars TC float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n2; i++)
		{
			for (int j = 1; j < n1; j++)
			{
				fprintf(fd, "%f ", tnCon[i][j][k]);
			}
		}
	}

	fclose(fd);
}
