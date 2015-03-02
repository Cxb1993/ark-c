#include "init.h"

void Input(Options opt)
{
	// geometry index
	l = opt.index_geometry;

	// total number of grid nodes along the x1 axis
	n1_g = opt.n1g;
	// total number of grid nodes along the X2 axis
	n2_g = opt.n2g;
	// total number of grid nodes along the X3 axis
	n3_g = opt.n3g;

	// number of grid nodes along the x1 axis to 1 processor
	n1 = n1_g/1;
	// number of grid nodes along the X2 axis to 1 processor
	n2 = n2_g/1;
	// number of grid nodes along the X3 axis to 1 processor
	n3 = n3_g/1;

	// coordinates of west plane
	x1_w = 0.;
	// coordinates of east plane
	x1_e = 2.;

	// coordinates of south plane
	x2_s = 0.;
	// coordinates of north plane
	x2_n = 2. * M_PI;

	// coordinates of bottom plane
	x3_b = 0.;
	// coordinates of top plane
	x3_t = 4. * M_PI;

	// total number of steps
	nStop = opt.nStop;
	// print interval
	nPrint = opt.nPrint;

	// Courant number
	CFL = opt.cfl;


	// kinematic viscosity
	VIS = 0.5/50.;  // 0.5
	// initial temperature
	t0 = 1.;

	// initial velocity along the x1 axis
	u10 = 0.;
	// initial velocity along the X2 axis
	u20 = 0.;
	// initial velocity along the X3 axis
	u30 = 1.;
	// sound velocity
	sound = 10.;

	// pressure on the top plane
	pOutlet = 0;
	// velocity on the bottom plane along the X3 axis
	u3Inlet = u30;
	// velocity on the bottom plane along the X2 axis
	u2Inlet = u20;
	// velocity on the bottom plane along the x1 axis
	u1Inlet = u10;
	// temperature on the bottom plane
	tInlet = t0;
	// unperturbed density of the liquid
	ro0_g = 1.;
	// unperturbed density of the borders material
	ro0_s = 100000000.;

	// #####################################################
	// 				block of arrays allocation
	// #####################################################

	// coordinates of grid nodes along the all of the axises
	x1 = calloc(n1 + 2, sizeof(double));
	x2 = calloc(n2 + 2, sizeof(double));
	x3 = calloc(n3 + 2, sizeof(double));

	// variables on the current time step
	roCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	tCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	u1Con = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	u2Con = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	u3Con = allocate3D(n2 + 1, n1 + 1, n3 + 1);

	// variables on the next time step
	ronCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	tnCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	u1nCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	u2nCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	u3nCon = allocate3D(n2 + 1, n1 + 1, n3 + 1);

	allocateForces(n1, n2, n3);

	// variables perpendicular to the axis x1
	ro1 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	t1 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u11 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u21 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u31 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	p1 = allocate3D(n2 + 2, n1 + 2, n3 + 2);

	// variables perpendicular to the axis X2
	ro2 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	t2 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u12 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u22 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u32 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	p2 = allocate3D(n2 + 2, n1 + 2, n3 + 2);

	// variables perpendicular to the axis X3
	ro3 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	t3 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u13 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u23 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	u33 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	p3 = allocate3D(n2 + 2, n1 + 2, n3 + 2);

	// NMAX = MAX(n1, n2, n3)
	int nmax;
	nmax = n1 < n2 ? n2 : n1;
	nmax = nmax < n3 ? n3 : nmax;

	// additional buffers for phase 2
	rBuf	= calloc((nmax+2), sizeof(double));
	qBuf	= calloc((nmax+1), sizeof(double));
	tfBuf	= calloc((nmax+2), sizeof(double));
	tbBuf	= calloc((nmax+2), sizeof(double));
	u2fBuf	= calloc((nmax+2), sizeof(double));
	u2bBuf	= calloc((nmax+2), sizeof(double));
	u3fBuf	= calloc((nmax+2), sizeof(double));
	u3bBuf	= calloc((nmax+2), sizeof(double));

	allocateStress(n1, n2, n3);
}


void InitializeData()
{
//	double	ALFA0 = 0.204,
//			BETA = 0.3,
//			R0 = 0.05;

	// grid step along the x1 axis
	dx1=(x1_e-x1_w)/(n1-1);
	dx2=(x2_n-x2_s)/(n2-1);
	dx3=(x3_t-x3_b)/(n3-1);

	x1[0] = x1_w - dx1;
	x2[0] = x2_s - dx2;
	x3[0] = x3_b - dx3;

	// #####################################################
	//			block of arrays initialization
	// #####################################################

	// along the X1 axis
	for (int j = 0; j < n1 + 1 ; ++j) {
		x1[j + 1] = x1[j] + dx1;
	}

	// along the X2 axis
	for (int i = 0; i < n2 + 1; ++i) {
		x2[i + 1] = x2[i] + dx2;
	}

	// along the X3 axis
	for (int k = 0; k < n3 + 1; ++k) {
		x3[k + 1] = x3[k] + dx3;
	}

	for (int i = 0; i < n2 + 1; ++i) {
		for (int j = 0; j < n1 + 1; ++j) {
			for (int k = 0; k < n3 + 1; ++k) {
				u1Con[i][j][k] = u1nCon[i][j][k] = u10;
				u2Con[i][j][k] = u2nCon[i][j][k] = u20;
				u3Con[i][j][k] = u3nCon[i][j][k] = u30;
				roCon[i][j][k] = ronCon[i][j][k] = ro0_g;
				tCon[i][j][k] = tnCon[i][j][k] = t0;
			}
		}
	}

	for (int i = 0; i < n2 + 2; ++i) {
		for (int j = 0; j < n1 + 2; ++j) {
			for (int k = 0; k < n3 + 2; ++k) {
				p1[i][j][k] = p2[i][j][k] = p3[i][j][k] = 0.;
				ro1[i][j][k] = ro2[i][j][k] = ro3[i][j][k] = ro0_g;
				u11[i][j][k] = u12[i][j][k] = u13[i][j][k] = u10;
				u21[i][j][k] = u22[i][j][k] = u23[i][j][k] = u20;
				u31[i][j][k] = u32[i][j][k] = u33[i][j][k] = u30;
				t1[i][j][k] = t2[i][j][k] = t3[i][j][k] = t0;
			}
		}
	}
}

double ***allocate3D(int n1, int n2, int n3)
{
	double ***arr;

	arr = calloc(n1, sizeof(double**));

	for (int i = 0; i < n1; ++i) {
		arr[i] = calloc(n2, sizeof(double*));
		for (int j = 0; j < n2; ++j) {
			arr[i][j] = calloc(n3, sizeof(double));
		}
	}
	return arr;
}
