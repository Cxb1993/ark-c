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
void WriteDataParaView();
void FreeMemory();

double ***allocate3D(int, int, int);
void deallocate3D(double***, int, int);

int main (int argc, char** argv)
{
	Input();
	InitializeData();

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
		UseForces();
		TIME += 0.5*dt;

		if (nStep % nPrint == 0) WriteDataParaView();
		printf("step: %d dt:%E\n", nStep, dt);
	} while (nStep < nStop);
	FreeMemory();
    return 0;
}

void Input()
{
	// geometry index
	l = 1;

	// total number of grid nodes along the x1 axis
	n1_g = 16;
	// total number of grid nodes along the X2 axis
	n2_g = 16;
	// total number of grid nodes along the X3 axis
	n3_g = 32;

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
	x2_n = 2. * PI;

	// coordinates of bottom plane
	x3_b = 0.;
	// coordinates of top plane
	x3_t = 4. * PI;

	// total number of steps
	nStop = 100;
	// print interval
	nPrint = 100;

	// Courant number
	CFL = 0.2;

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
	x1 = calloc(n1+2, sizeof(double));
	x2 = calloc(n2+2, sizeof(double));
	x3 = calloc(n3+2, sizeof(double));

	// variables on the current time step
	roCon	= allocate3D(n1, n2, n3);
	tCon	= allocate3D(n1, n2, n3);
	u1Con	= allocate3D(n1, n2, n3);
	u2Con	= allocate3D(n1, n2, n3);
	u3Con	= allocate3D(n1, n2, n3);

	// variables on the next time step
	ronCon	= allocate3D(n1, n3, n3);
	tnCon	= allocate3D(n1, n2, n3);
	u1nCon	= allocate3D(n1, n2, n3);
	u2nCon	= allocate3D(n1, n2, n3);
	u3nCon	= allocate3D(n1, n2, n3);

	// variables perpendicular to the axis x1
	ro1	= allocate3D(n1+1, n2, n3);
	t1	= allocate3D(n1+1, n2, n3);
	u11	= allocate3D(n1+1, n2, n3);
	u21	= allocate3D(n1+1, n2, n3);
	u31	= allocate3D(n1+1, n2, n3);
	p1	= allocate3D(n1+1, n2, n3);

	// variables perpendicular to the axis X2
	ro2	= allocate3D(n1, n2+1, n3);
	t2	= allocate3D(n1, n2+1, n3);
	u12	= allocate3D(n1, n2+1, n3);
	u22	= allocate3D(n1, n2+1, n3);
	u32	= allocate3D(n1, n2+1, n3);
	p2	= allocate3D(n1, n2+1, n3);

	// variables perpendicular to the axis X3
	ro3	= allocate3D(n1, n2, n3+1);
	t3	= allocate3D(n1, n2, n3+1);
	u13	= allocate3D(n1, n2, n3+1);
	u23	= allocate3D(n1, n2, n3+1);
	u33	= allocate3D(n1, n2, n3+1);
	p3	= allocate3D(n1, n2, n3+1);

	// forces
	f1 = allocate3D(n1, n2, n3);
	f2 = allocate3D(n1, n2, n3);
	f3 = allocate3D(n1, n2, n3);

	// get into NMAX maxim of n1, n2, n3
	int nmax = (n1 > n2 ? n1 : n2);
	nmax = nmax > n3 ? nmax : n3;

	// additional buffers for phase 2
	rBuf	= calloc((nmax+2), sizeof(double));
	qBuf	= calloc((nmax+1), sizeof(double));
	tfBuf	= calloc((nmax+2), sizeof(double));
	tbBuf	= calloc((nmax+2), sizeof(double));
	u2fBuf	= calloc((nmax+2), sizeof(double));
	u2bBuf	= calloc((nmax+2), sizeof(double));
	u3fBuf	= calloc((nmax+2), sizeof(double));
	u3bBuf	= calloc((nmax+2), sizeof(double));

	// friction stress
	sigm11 = allocate3D(n1, n2, n3);
	sigm21 = allocate3D(n1, n2, n3);
	sigm31 = allocate3D(n1, n2, n3);

	sigm12 = allocate3D(n1, n2, n3);
	sigm22 = allocate3D(n1, n2, n3);
	sigm32 = allocate3D(n1, n2, n3);

	sigm13 = allocate3D(n1, n2, n3);
	sigm23 = allocate3D(n1, n2, n3);
	sigm33 = allocate3D(n1, n2, n3);
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

	// block of arrays initialization
	for (int i = 0; i <= n1 ; ++i) {
		x1[i+1] = x1[i] + dx1;
	}

	for (int j = 0; j <= n2 ; ++j) {
		x2[j +1] = x2[j] + dx2;
	}

	for (int k = 0; k <= n3 ; ++k) {
		x3[k +1] = x3[k] + dx3;
	}

	for (int i = 0; i <= n1; ++i) {
		for (int j = 0; j <= n2; ++j) {
			for (int k = 0; k <= n3; ++k) {
				u1Con[i][j][k] = u1nCon[i][j][k] = u10;
				u2Con[i][j][k] = u2nCon[i][j][k] = u20;
				u3Con[i][j][k] = u3nCon[i][j][k] = u30;
				roCon[i][j][k] = ronCon[i][j][k] = ro0_g;
				tCon[i][j][k] = tnCon[i][j][k] = t0;

				p1[i][j][k] = p2[i][j][k] = p3[i][j][k] = 0.;
				ro1[i][j][k] = ro2[i][j][k] = ro3[i][j][k] = ro0_g;
				u11[i][j][k] = u12[i][j][k] = u13[i][j][k] = u10;
				u21[i][j][k] = u22[i][j][k] = u23[i][j][k] = u20;
				u31[i][j][k] = u32[i][j][k] = u33[i][j][k] = u30;
				t1[i][j][k] = t2[i][j][k] = t3[i][j][k] = t0;
			}

		}
	}

	for (int j = 0; j <= n2; ++j) {
		for (int k = 0; k <= n3; ++k) {
			p1[n1+1][j][k] = 0.;
			ro1[n1+1][j][k] = ro0_g;
			u11[n1+1][j][k] = u10;
			u21[n1+1][j][k] = u20;
			u31[n1+1][j][k] = u30;
			t1[n1+1][j][k] = t0;
		}
	}

	for (int i = 0; i <= n1; ++i) {
		for (int k = 0; k <= n3; ++k) {
			p2[i][n2+1][k] = 0.;
			ro2[i][n2+1][k] = ro0_g;
			u12[i][n2+1][k] = u10;
			u22[i][n2+1][k] = u20;
			u32[i][n2+1][k] = u30;
			t2[i][n2+1][k] = t0;
		}
	}

	for (int i = 0; i <= n1; ++i) {
		for (int j = 0; j <= n2; ++j) {
			p3[i][j][n3+1] = 0.;
			ro3[i][j][n3+1] = ro0_g;
			u13[i][j][n3+1] = u10;
			u23[i][j][n3+1] = u20;
			u33[i][j][n3+1] = u30;
			t3[i][j][n3+1] = t0;
		}
	}

}

void TimeStepSize()
{
	double x1c, x1l, u1c, u2c, u3c, dtu1, dtu2, dtu3, dtu, dtv1, dtv2, dtv3, dtv;

	dt = pow(10, 8);

	for (int i = 1; i < n1; ++i) {
		for (int j = 1; j < n2; ++j) {
			for (int k = 1; k < n3; ++k) {
				x1c = (x1[i+1] + x1[i])/2;
				x1l = 1+(l-1)*(x1c-1);

				u1c = u1Con[i][j][k];
				u2c = u2Con[i][j][k];
				u3c = u3Con[i][j][k];

				dtu1 = CFL*dx1/(sound + fabs(u1c));
				dtu2 = CFL*x1l*dx2/(sound + fabs(u2c));
				dtu3 = CFL*dx3/(sound + fabs(u3c));

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
				double dttemp = dtu > dtv ? dtv : dtu;
				if (dttemp < dt) dt = dttemp;
			}
		}
	}
}

void Phase1()
{
	// initialization
	for (int i = 1; i < n2; i++)
	{
		for (int j = 1; j < n1; j++)
		{
			for (int k = 1; k < n3; k++)
			{
				u1Con[i][j][k] = u1nCon[i][j][k];
				u1Con[i][j][k] = u1nCon[i][j][k];
				u1Con[i][j][k] = u1nCon[i][j][k];
				roCon[i][j][k] = ronCon[i][j][k];
				tCon[i][j][k] = tnCon[i][j][k];
			}
		}
	}

	// geometric characteristics of the computational cell
	double ds1, ds2, ds3, dvc, xle, xlw, xlt, xlb, alfa;

			// velocity, density, temperature and pressure on the eastern plane
	double	u1_e, u2_e, u3_e, roe, te, pe,
			// velocity, density , temperature and pressure on the western plane
			u1_w, u2_w, u3_w, row, tw, pw,
			// velocity, density , temperature and pressure on the northern plane
			u1_n, u2_n, u3_n, ron, tn, pn,
			// velocity, density , temperature and pressure on the southern plane
			u1_s, u2_s, u3_s, ros, ts, ps,
			// velocity, density , temperature and pressure on the top plane
			u1_t, u2_t, u3_t, rot, tt, pt,
			// velocity, density , temperature and pressure on the bottom plane
			u1_b, u2_b, u3_b, rob, tb, pb,
			// velocity, density and temperature in the cell center
			u1_c, u2_c, u3_c, roc, tc,
			// velocity, density and temperature in the cell center on the next time step
			u1_cn, u2_cn, u3_cn, rocn, tcn;

	// plane squares
	ds1 = dx2*dx3;
	ds2 = dx1*dx3; 
	ds3 = dx1*dx2;

	for (int i = 1; i < n1; i++)
	{
		for (int j = 1; j < n2; j++)
		{
			for (int k = 1; k < n3; k++)
			{
				// geometric factor of cylindricity
				xle = 1 + (l - 1)*(x1[i] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l - 1)*(x1[i - 1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);
				// geometric factor of cylindricity
				xlb = xlt;
				// cell volume
				dvc = 0.5*(xle + xlw)*dx1*dx2*dx3;
				// geometric parameter
				alfa = dt*(l - 1)*dx1*dx2*dx3 / (2 * dvc);

				// #########################################################
				//	get velocity, density , temperature and pressure values
				// #########################################################

				// east plane
				u1_e = u11[i + 1][j][k];
				u2_e = u21[i + 1][j][k];
				u3_e = u31[i + 1][j][k];
				roe = ro1[i + 1][j][k];
				te = t1[i + 1][j][k];
				pe = p1[i + 1][j][k];
				// west plane
				u1_w = u11[i][j][k];
				u2_w = u21[i][j][k];
				u3_w = u31[i][j][k];
				row = ro1[i][j][k];
				tw = t1[i][j][k];
				pw = p1[i][j][k];

				// north plane
				u1_n = u12[i][j + 1][k];
				u2_n = u22[i][j + 1][k];
				u3_n = u32[i][j + 1][k];
				ron = ro2[i][j + 1][k];
				tn = t2[i][j + 1][k];
				pn = p2[i][j + 1][k];
				// south plane
				u1_s = u12[i][j][k];
				u2_s = u22[i][j][k];
				u3_s = u32[i][j][k];
				ros = ro2[i][j][k];
				ts = t2[i][j][k];
				ps = p2[i][j][k];

				// top plane
				u1_t = u13[i][j][k + 1];
				u2_t = u23[i][j][k + 1];
				u3_t = u33[i][j][k + 1];
				rot = ro3[i][j][k + 1];
				tt = t3[i][j][k + 1];
				pt = p3[i][j][k + 1];
				// bottom plane
				u1_b = u13[i][j][k];
				u2_b = u23[i][j][k];
				u3_b = u33[i][j][k];
				rob = ro3[i][j][k];
				tb = t3[i][j][k];
				pb = p3[i][j][k];

				// cell center
				u1_c = u1Con[i][j][k];
				u2_c = u2Con[i][j][k];
				u3_c = u3Con[i][j][k];
				roc = roCon[i][j][k];
				tc = tCon[i][j][k];
				
				// #####################################################
				//					new values evaluating
				// #####################################################

				// new density
				rocn = (roc*dvc - 0.5*dt*((xle*roe*u1_e - xlw*row*u1_w)*ds1 +
					(ron*u2_n - ros*u2_s)*ds2 + (xlt*rot*u3_t - xlb*rob*u3_b)*ds3)) / dvc;
				
				// new conservative velocity along the x1 axis
				double U1CP = (roc*u1_c*dvc - 0.5*dt*((ron*u2_n*u1_n - ros*u2_s*u1_s)*ds2 +
					(xlt*rot*u3_t*u1_t - xlb*rob*u3_b*u1_b)*ds3 +
					(xle*(roe*u1_e*roe*u1_e + pe) - xlw*(row*u1_w*row*u1_w + pw))*ds1
					- 0.5*(l - 1)*(pe + pw)*dx1*dx2*dx3)) / (dvc*rocn);
				// new conservative velocity along the X2 axis
				double U2CP = U2CP = (roc*u2_c*dvc - 0.5*dt*((xle*roe*u1_e*u2_e - xlw*row*u1_w*u2_w)*ds1 +
					((ron*u2_n*ron*u2_n + pn) - (ros*u2_s*ros*u2_s + ps))*ds2 +
					(xlt*rot*u3_t*u2_t - xlb*rob*u3_b*u2_b)*ds3)) / (dvc*rocn);

				// take into account of centrifugal and Coriolis forces
				u1_cn = (U1CP - alfa*u2_c*U1CP) / (1 + (alfa*u2_c)*(alfa*u2_c));
				u2_cn = U2CP - alfa*u2_c*u1_cn;

				// new conservative velocity along the X3 axis
				u3_cn = (roc*u3_c*dvc - 0.5*dt*((xle*roe*u1_e*u3_e - xlw*row*u1_w*u3_w)*ds1 +
					(ron*u2_n*u3_n - ros*u2_s*u3_s)*ds2 +
					(xlt*(rot*u3_t*rot*u3_t + pt) - xlb*(rob*u3_b*rob*u3_b + pb))*ds3)) / (dvc*rocn);

				// new temperature
				tcn = (roc*tc*dvc - 0.5*dt*((xle*roe*te*u1_e - xlw*row*tw*u1_w)*ds1 +
					(ron*tn*u2_n - ros*ts*u2_s)*ds2 +
					(xlt*rot*tt*u3_t - xlb*rob*tb*u3_b)*ds3)) / (dvc*rocn);

				// finally
				u1nCon[i][j][k] = u1_cn;
				u2nCon[i][j][k] = u2_cn;
				u3nCon[i][j][k] = u3_cn;
				ronCon[i][j][k] = rocn;
				tnCon[i][j][k] = tcn;
			}
		}
	}

	// periodicity conditions
	for (int i = 1; i <= n1; ++i)
	{
		for (int k = 1; k <= n3; ++k)
		{
			// periodicity condition on the north plane
			u1nCon[i][n2][k] = u1nCon[i][1][k];
			u2nCon[i][n2][k] = u2nCon[i][1][k];
			u3nCon[i][n2][k] = u3nCon[i][1][k];
			ronCon[i][n2][k] = ronCon[i][1][k];
			tnCon[i][n2][k] = tnCon[i][1][k];
			// periodicity condition on the south plane
			u1nCon[i][0][k] = u1nCon[i][n2 - 1][k];
			u2nCon[i][0][k] = u2nCon[i][n2 - 1][k];
			u3nCon[i][0][k] = u3nCon[i][n2 - 1][k];
			ronCon[i][0][k] = ronCon[i][n2 - 1][k];
			tnCon[i][0][k] = tnCon[i][n2 - 1][k];
		}
	}	

	// no-slip conditions
	for (int j = 1; j < n2; ++j)
	{
		for (int k = 1; k < n3; ++k)
		{
			// no-slip consition on the west plane
			u1nCon[0][j][k] = 0.;
			u2nCon[0][j][k] = 0.;
			u3nCon[0][j][k] = 0.;
			ronCon[0][j][k] = ronCon[1][j][k];
			tnCon[0][j][k] = t0;

			// no-slip consition on the east plane
			u1nCon[n1][j][k] = 0.;
			u2nCon[n1][j][k] = 0.;
			u3nCon[n1][j][k] = 0.;
			ronCon[n1][j][k] = ronCon[n1 - 1][j][k];
			tnCon[n1][j][k] = t0;
		}
	}
}

void Phase2()
{
	double u1f, u1b, u1cn, u1c,
		u2f, u2fn, u2b, u2bn, u2cn, u2c,
		u3f, U3FN, u3b, U3BN, U3CN, u3c,
		rof, rob, rocn, roc,
		tf, tfn, tb, tbn, tcn, tc,
		pf, pb, pcn, pc,
		rf, rfn, rb, rcn, rc,
		qf, qb, qbn, QCN, qc;

	double gr, gt, gu2, gu3,
		gq;

	double rmax, rmin, qmax, qmin, tmax, tmin, U2MAX, U2MIN, u3max, u3min;

	double ro0b, ro0f, qn, pn, rn, ron, tn, un, u2n, u3n, ucf, ucb;

	// first local invariants for the interior faces puts in the buffer arrays, bypass on the center of the cell
	// then by taking into account the boundary condition calculates extreme elements of the buffers
	// and only then calculates the flow variables

	// flow variables calculation on DS1 faces orthogonal x1 axis 

	// bypassing along the x1 axis

	// only interior faces !
	for (int k = 1; k < n3; k++)
	{
		for (int j = 1; j < n2; j++)
		{
			for (int i = 1; i < n1; i++)
			{
				u1f = u11[i + 1][j][k];
				u1b = u11[i][j][k];
				u1cn = u1nCon[i][j][k];
				u1c = u1Con[i][j][k];

				u2f = u21[i + 1][j][k];
				u2b = u21[i][j][k];
				u2cn = u2nCon[i][j][k];
				u2c = u2Con[i][j][k];

				u3f = u31[i + 1][j][k];
				u3b = u31[i][j][k];
				U3CN = u3nCon[i][j][k];
				u3c = u3Con[i][j][k];

				rof = ro1[i + 1][j][k];
				rob = ro1[i][j][k];
				rocn = roCon[i][j][k];
				roc = roCon[i][j][k];

				tf = t1[i + 1][j][k];
				tb = t1[i][j][k];
				tcn = tnCon[i][j][k];
				tc = tCon[i][j][k];

				pf = p1[i + 1][j][k];
				pb = p1[i][j][k];
				pcn = sound*sound*(rocn - ro0_g);
				pc = sound*sound*(roc - ro0_g);

				// invariant calculation

				rf = u1f + pf / (ro0_g * sound);
				rb = u1b + pb / (ro0_g * sound);
				rcn = u1cn + pcn / (ro0_g * sound);
				rc = u1c + pc / (ro0_g * sound);

				rfn = 2 * rcn - rb;

				qf = u1f - pf / (ro0_g * sound);
				qb = u1b - pb / (ro0_g * sound);
				QCN = u1cn - pcn / (ro0_g * sound);
				qc = u1c - pc / (ro0_g * sound);

				qbn = 2 * QCN - qf;

				tfn = 2 * tcn - tb;
				tbn = 2 * tcn - tf;

				u2fn = 2 * u2cn - u2b;
				u2bn = 2 * u2cn - u2f;

				U3FN = 2 * U3CN - u3b;
				U3BN = 2 * U3CN - u3f;

				// the permissible range of changes
				gr = 2 * (rcn - rc) / dt + (u1cn + sound)*(rf - rb) / dx1;
				gq = 2 * (rcn - rc) / dt + (u1cn - sound)*(qf - qb) / dx1;

				gt = 2 * (tcn - tc) / dt + u1cn*(tf - tb) / dx1;
				gu2 = 2 * (u2cn - u2c) / dt + u1cn*(u2f - u2b) / dx1;
				gu3 = 2 * (U3CN - u3c) / dt + u1cn*(u3f - u3b) / dx1;

				// RMAX=MAX(RF,RC,RB) +dt*GR
				rmax = rf > rc ? rf : rc;
				rmax = rmax > rb ? rmax : rb;
				rmax += dt*gr;

				// RMIN=MIN(RF,RC,RB) +dt*GR
				rmin = rf < rc ? rf : rc;
				rmin = rmin < rb ? rmin : rb;
				rmin += dt*gr;

				// QMAX=MAX(QF,QC,QB) +dt*GQ
				qmax = qf > qc ? qf : qc;
				qmax = qmax > qb ? qmax : qb;
				qmax += dt*gq;

				// QMIN=MIN(QF,QC,QB) +dt*GQ
				qmin = qf < qc ? qf : qc;
				qmin = qmin < qb ? qmin : qb;
				qmin += dt*gq;

				// TMAX=MAX(TF,TC,TB) +dt*GT
				tmax = tf > tc ? tf : tc;
				tmax = tmax > tb ? tmax : tb;
				tmax += dt*gt;

				// TMIN=MIN(TF,TC,TB) +dt*GT
				tmin = tf < tc ? tf : tc;
				tmin = tmin < tb ? tmin : tb;
				tmin += dt*gt;

				// U2MAX=MAX(U2F,U2C,U2B) +dt*GU2
				U2MAX = u2f > u2c ? u2f : u2c;
				U2MAX = U2MAX > u2b ? U2MAX : u2b;
				U2MAX += dt*gu2;

				// U2MIN=MIN(U2F,U2C,U2B) +dt*GU2
				U2MIN = u2f < u2c ? u2f : u2c;
				U2MIN = U2MIN < u2b ? U2MIN : u2b;
				U2MIN += dt*gu2;

				// U3MAX=MAX(U3F,U3C,U3B) +dt*GU3
				u3max = u3f > u3c ? u3f : u3c;
				u3max = u3max > u3b ? u3max : u3b;
				u3max += dt*gu3;

				// U3MIN=MIN(U3F,U3C,U3B) +dt*GU3 
				u3min = u3f < u3c ? u3f : u3c;
				u3min = u3min < u3b ? u3min : u3b;
				u3min += dt*gu3;

				// invariants correction
				if (rfn > rmax) rfn = rmax;
				if (rfn < rmin) rfn = rmin;

				if (qbn > qmax) qbn = qmax;
				if (qbn < qmin) qbn = qmin;

				if (tfn > tmax) tfn = tmax;
				if (tfn < tmin) tfn = tmin;

				if (tbn > tmax) tbn = tmax;
				if (tbn < tmin) tbn = tmin;

				if (u2fn > U2MAX) u2fn = U2MAX;
				if (u2fn < U2MIN) u2fn = U2MIN;

				if (u2bn > U2MAX) u2bn = U2MAX;
				if (u2bn < U2MIN) u2bn = U2MIN;

				if (U3FN > u3max) U3FN = u3max;
				if (U3FN < u3min) U3FN = u3min;

				if (U3BN > u3max) U3BN = u3max;
				if (U3BN < u3min) U3BN = u3min;

				// put invariants to buffers
				rBuf[i + 1] = rfn;
				qBuf[i] = qbn;
				tfBuf[i + 1] = tfn;
				tbBuf[i] = tbn;
				u2fBuf[i + 1] = u2fn;
				u2bBuf[i] = u2bn;
				u3fBuf[i + 1] = U3FN;
				u3bBuf[i] = U3BN;
			}

			// boundary conditions along the x1 axis
			// assignment of boundary invatiants and add them to the buffer arrays

			// periodicity conditions
			rBuf[1] = rBuf[n1];
			tfBuf[1] = tfBuf[n1];
			u2fBuf[1] = u2fBuf[n1];
			u3fBuf[1] = u3fBuf[n1];

			// periodicity conditions
			qBuf[n1] = qBuf[1];
			tbBuf[n1] = tbBuf[1];
			u2bBuf[n1] = u2bBuf[1];
			u3bBuf[n1] = u3bBuf[1];

			// no-slip conditions
			// i == 1
			ro0b = ronCon[n1 - 1][j][k];
			ro0f = ronCon[1][j][k];

			qn = qBuf[1];
			un = 0;
			pn = -qn*sound*ro0_g;
			ron = (ro0_g + pn / (sound*sound));

			tn = t0;
			u2n = 0;
			u3n = 0;

			p1[1][j][k] = pn;
			u11[1][j][k] = un;
			ro1[1][j][k] = ron;
			t1[1][j][k] = tn;
			u21[1][j][k] = u2n;
			u31[1][j][k] = u3n;


			// i == n1
			rn = rBuf[n1];

			un = 0;
			pn = rn*sound*ro0_g;
			ron = (ro0_g + pn / (sound*sound));

			tn = t0;
			u2n = 0;
			u3n = 0;

			p1[n1][j][k] = pn;
			u11[n1][j][k] = un;
			ro1[n1][j][k] = ron;
			t1[n1][j][k] = tn;
			u21[n1][j][k] = u2n;
			u31[n1][j][k] = u3n;

			// the flow variables calculations
			for (int i = 2; i < n1; i++)
			{
				ro0b = ronCon[i - 1][j][k];
				ro0f = ronCon[i][j][k];

				rn = rBuf[i];
				qn = qBuf[i];

				pn = (rn - qn)*sound*ro0_g / 2;
				un = (rn + qn) / 2;

				ron = (ro0_g + pn / (sound*sound));

				ucf = u1nCon[i][j][k];
				ucb = u1nCon[i - 1][j][k];

				if (ucf >= 0 && ucb > 0)
				{
					tn = tfBuf[i];
					u2n = u2fBuf[i];
					u3n = u3fBuf[i];
				}
				else if (ucf <= 0 && ucb <= 0)
				{
					tn = tbBuf[i];
					u2n = u2bBuf[i];
					u3n = u3bBuf[i];
				}
				else if (ucb >= 0 && ucf <= 0) 
				{
					if (ucb > -ucf) 
					{
						tn = tfBuf[i];
						u2n = u2fBuf[i];
						u3n = u3fBuf[i];
					}
					else 
					{
						tn = tbBuf[i];
						u2n = u2bBuf[i];
						u3n = u3bBuf[i];
					}
				}
				else
				{
					if (ucb <= 0 && ucf >= 0)
					{
						tn = tnCon[i][j][k] + tnCon[i - 1][j][k] - t1[i][j][k];
						u2n = u2nCon[i][j][k] + u2nCon[i - 1][j][k] - u21[i][j][k];
						u3n = u3nCon[i][j][k] + u3nCon[i - 1][j][k] - u31[i][j][k];
					}
				}

				p1[i][j][k] = pn;
				u11[i][j][k] = un;
				ro1[i][j][k] = ron;
				t1[i][j][k] = tn;
				u21[i][j][k] = u2n;
				u31[i][j][k] = u3n;
			}

			// the flow variable calculations on the east border
			p1[1][j][k] = p1[n1][j][k];
			u11[1][j][k] = u11[n1][j][k];
			ro1[1][j][k] = ro1[n1][j][k];
			t1[1][j][k] = t1[n1][j][k];
			u21[1][j][k] = u21[n1][j][k];
			u31[1][j][k] = u31[n1][j][k];
		}
	}

	// flow variables calculation on DS2 faces orthogonal X2 axis 

	// bypassing along the X2 axis

	double xle, xlw, xls;

	double u1fn, u1bn;

	double gu1;

	double u1max, u1min, u1n;

	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n1; i++)
		{
			xle = 1 + (l - 1)*(x1[i + 1] - 1);
			xlw = 1 + (l - 1)*(x1[i] - 1);
			xls = 0.5*(xle + xlw);
			
			for (int j = 0; j <= n2; j++)
			{
				u2f = u22[i][j + 1][k];
				u2b = u22[i][j][k];
				u2cn = u2nCon[i][j][k];
				u2c = u2Con[i][j][k];

				u1f = u12[i][j + 1][k];
				u1b = u12[i][j][k];
				u1cn = u1nCon[i][j][k];
				u1c = u1Con[i][j][k];

				u3f = u32[i][j + 1][k];
				u3b = u32[i][j][k];
				U3CN = u3nCon[i][j][k];
				u3c = u3Con[i][j][k];

				rof = ro2[i][j + 1][k];
				rob = ro2[i][j][k];
				rocn = ronCon[i][j][k];
				roc = roCon[i][j][k];

				tf = t2[i][j + 1][k];
				tb = t2[i][j][k];
				tcn = tnCon[i][j][k];
				tc = tCon[i][j][k];

				pf = p2[i][j + 1][k];
				pb = p2[i][j][k];
				pcn = sound*sound*(rocn - ro0_g);
				pc = sound*sound*(roc - ro0_g);

				// invariant calculation
				rf = u2f + pf / (ro0_g*sound);
				rb = u2b + pb / (ro0_g*sound);
				rcn = u2cn + pcn / (ro0_g*sound);
				rc = u2c + pc / (ro0_g*sound);

				rfn = 2 * rcn - rb;

				qf = u2f - pf / (ro0_g*sound);
				qb = u2b - pb / (ro0_g*sound);
				QCN = u2cn - pcn / (ro0_g*sound);
				qc = u2c - pc / (ro0_g*sound);

				qbn = 2 * QCN - qf;

				tfn = 2 * tcn - tb;
				tbn = 2 * tcn - tf;

				u1fn = 2 * u1cn - u1b;
				u1bn = 2 * u1cn - u1f;

				U3FN = 2 * U3CN - u3b;
				U3BN = 2 * U3CN - u3f;

				// the permissible range of changes
				gr = 2 * (rcn - rc) / dt + (u2cn + sound)*(rf - rb) / dx2 / xls;
				gq = 2 * (QCN - qc) / dt + (u2cn - sound)*(qf - qb) / dx2 / xls;
				gt = 2 * (tcn - tc) / dt + u2cn*(tf - tb) / dx2 / xls;
				gu1 = 2 * (u1cn - u1c) / dt + u2cn*(u1f - u1b) / dx2 / xls;
				gu3 = 2 * (U3CN - u3c) / dt + u2cn*(u3f - u3b) / dx2 / xls;

				// RMAX=MAX(RF,RC,RB) +dt*GR
				rmax = rf > rc ? rf : rc;
				rmax = rmax > rb ? rmax : rb;
				rmax += dt*gr;

				// RMIN=MIN(RF,RC,RB) +dt*GR
				rmin = rf < rc ? rf : rc;
				rmin = rmin < rb ? rmin : rb;
				rmin += dt*gr;

				// QMAX=MAX(QF,QC,QB) +dt*GQ
				qmax = qf > qc ? qf : qc;
				qmax = qmax > qb ? qmax : qb;
				qmax += dt*gq;

				// QMIN=MIN(QF,QC,QB) +dt*GQ
				qmin = qf < qc ? qf : qc;
				qmin = qmin < qb ? qmin : qb;
				qmin += dt*gq;

				// TMAX=MAX(TF,TC,TB) +dt*GT
				tmax = tf > tc ? tf : tc;
				tmax = tmax > tb ? tmax : tb;
				tmax += dt*gt;

				// TMIN=MIN(TF,TC,TB) +dt*GT
				tmin = tf < tc ? tf : tc;
				tmin = tmin < tb ? tmin : tb;
				tmin += dt*gt;

				// U1MAX=MAX(U1F,U1C,U1B) +dt*GU1
				u1max = u1f > u1c ? u1f : u1c;
				u1max = u1max > u1b ? u1max : u1b;
				u1max += dt*gu1;

				// U1MIN=MIN(U1F,U1C,U1B) +dt*GU1
				u1min = u1f < u1c ? u1f : u1c;
				u1min = u1min < u1b ? u1min : u1b;
				u1min += dt*gu1;

				// U3MAX=MAX(U3F,U3C,U3B) +dt*GU3
				u3max = u3f > u3c ? u3f : u3c;
				u3max = u3max > u3b ? u3max : u3b;
				u3max += dt*gu3;

				// U3MIN=MIN(U3F,U3C,U3B) +dt*GU3 
				u3min = u3f < u3c ? u3f : u3c;
				u3min = u3min < u3b ? u3min : u3b;
				u3min += dt*gu3;

				// invariants correction
				if (rfn > rmax) rfn = rmax;
				if (rfn < rmin) rfn = rmin;

				if (qbn > qmax) qbn = qmax;
				if (qbn < qmin) qbn = qmin;

				if (tfn > tmax) tfn = tmax;
				if (tfn < tmin) tfn = tmin;

				if (tbn > tmax) tbn = tmax;
				if (tbn < tmin) tbn = tmin;

				if (u1fn > u1max) u1fn = u1max;
				if (u1fn < u1min) u1fn = u1min;

				if (u1bn > u1max) u1bn = u1max;
				if (u1bn < u1min) u1bn = u1min;

				if (U3FN > u3max) U3FN = u3max;
				if (U3FN < u3min) U3FN = u3min;

				if (U3BN > u3max) U3BN = u3max;
				if (U3BN < u3min) U3BN = u3min;

				// put invariants to buffers
				rBuf[j + 1] = rfn;
				qBuf[j] = qbn;
				tfBuf[j + 1] = tfn;
				tbBuf[j] = tbn;
				u2fBuf[j + 1] = u1fn;
				u2bBuf[j] = u1bn;
				u3fBuf[j + 1] = U3FN;
				u3bBuf[j] = U3BN;
			}

			// boundary conditions along the X2 axis
			// assignment of boundary invatiants and add them to the buffer arrays

			// periodicity conditions
			rBuf[1] = rBuf[n2];
			tfBuf[1] = tfBuf[n2];
			u2fBuf[1] = u2fBuf[n2];
			u3fBuf[1] = u3fBuf[n2];

			// periodicity conditions
			qBuf[n2] = qBuf[1];
			tbBuf[n2] = tbBuf[1];
			u2bBuf[n2] = u2bBuf[1];
			u3bBuf[n2] = u3bBuf[1];

			// the flow variables calculations
			for (int j = 1; j <= n2; j++)
			{
				ro0b = ronCon[i][j - 1][k];
				ro0f = ronCon[i][j][k];

				rn = rBuf[j];
				qn = qBuf[j];

				pn = (rn - qn)*sound*ro0_g / 2;
				un = (rn + qn) / 2;

				ron = (ro0_g + pn / (sound*sound));

				ucf = u2nCon[i][j][k];
				ucb = u2nCon[i][j - 1][k];

				if (ucf >= 0 && ucb > 0)
				{
					tn = tfBuf[j];
					u1n = u2fBuf[j];
					u3n = u3fBuf[j];
				}
				else if (ucf <= 0 && ucb <= 0)
				{
					tn = tbBuf[j];
					u1n = u2bBuf[j];
					u3n = u3bBuf[j];
				}
				else if (ucb >= 0 && ucf <= 0)
				{
					if (ucb > -ucf)
					{
						tn = tfBuf[j];
						u1n = u2fBuf[j];
						u3n = u3fBuf[j];
					}
					else
					{
						tn = tbBuf[j];
						u1n = u2bBuf[j];
						u3n = u3bBuf[j];
					}
				}
				else
				{
					if (ucb <= 0 && ucf >= 0)
					{
						tn = tnCon[i][j][k] + tnCon[i][j - 1][k] - t2[i][j][k];
						u1n = u1nCon[i][j][k] + u1nCon[i][j - 1][k] - u12[i][j][k];
						u3n = u3nCon[i][j][k] + u3nCon[i][j - 1][k] - u32[i][j][k];
					}
				}

				p2[i][j][k] = pn;
				u22[i][j][k] = un;
				ro2[i][j][k] = ron;
				t2[i][j][k] = tn;
				u12[i][j][k] = u1n;
				u32[i][j][k] = u3n;
			}

			// the flow variable calculations on the south border
			p2[i][1][k] = p2[i][n2][k];
			u22[i][1][k] = u22[i][n2][k];
			ro2[i][1][k] = ro2[i][n2][k];
			t2[i][1][k] = t2[i][n2][k];
			u12[i][1][k] = u12[i][n2][k];
			u32[i][1][k] = u32[i][n2][k];
		}
	}

	// flow variables calculation on DS3 faces orthogonal X3 axis 

	// bypassing along the X3 axis

	for (int j = 1; j < n2; j++)
	{
		for (int i = 1; i < n1; i++)
		{
			for (int k = 1; k < n3; k++)
			{
				u3f = u33[i][j][k + 1];
				u3b = u33[i][j][k];
				U3CN = u3nCon[i][j][k];
				u3c = u3Con[i][j][k];

				u2f = u23[i][j][k + 1];
				u2b = u23[i][j][k];
				u2cn = u2nCon[i][j][k];
				u2c = u2Con[i][j][k];

				u1f = u13[i][j][k + 1];
				u1b = u13[i][j][k];
				u1cn = u1nCon[i][j][k];
				u1c = u1Con[i][j][k];

				rof = ro3[i][j][k + 1];
				rob = ro3[i][j][k];
				rocn = ronCon[i][j][k];
				roc = roCon[i][j][k];

				tf = t3[i][j][k + 1];
				tb = t3[i][j][k];
				tcn = tnCon[i][j][k];
				tc = tCon[i][j][k];

				pf = p3[i][j][k + 1];
				pb = p3[i][j][k];
				pcn = sound*sound*(rocn - ro0_g);
				pc = sound*sound*(roc - ro0_g);

				// invariant calculation
				rf = u3f + pf / (ro0_g*sound);
				rb = u3b + pb / (ro0_g*sound);
				rcn = U3CN + pcn / (ro0_g*sound);
				rc = u3c + pc / (ro0_g*sound);

				rfn = 2 * rcn - rb;

				qf = u3f - pf / (ro0_g*sound);
				qb = u3b - pb / (ro0_g*sound);
				QCN = U3CN - pcn / (ro0_g*sound);
				qc = u3c - pc / (ro0_g*sound);

				qbn = 2 * QCN - qf;

				tfn = 2 * tcn - tb;
				tbn = 2 * tcn - tf;

				u2fn = 2 * u2cn - u2b;
				u2bn = 2 * u2cn - u2f;

				u1fn = 2 * u1cn - u1b;
				u1bn = 2 * u1cn - u1f;

				// the permissible range of changes
				gr = 2 * (rcn - rc) / dt + (U3CN + sound)*(rf - rb) / dx3;
				gq = 2 * (rcn - rc) / dt + (U3CN - sound)*(qf - qb) / dx3;

				gt = 2 * (tcn - tc) / dt + U3CN*(tf - tb) / dx3;
				gu2 = 2 * (u2cn - u2c) / dt + U3CN*(u2f - u2b) / dx3;
				gu1 = 2 * (u1cn - u1c) / dt + U3CN*(u1f - u1b) / dx3;

				// RMAX=MAX(RF,RC,RB) +dt*GR
				rmax = rf > rc ? rf : rc;
				rmax = rmax > rb ? rmax : rb;
				rmax += dt*gr;

				// RMIN=MIN(RF,RC,RB) +dt*GR
				rmin = rf < rc ? rf : rc;
				rmin = rmin < rb ? rmin : rb;
				rmin += dt*gr;

				// QMAX=MAX(QF,QC,QB) +dt*GQ
				qmax = qf > qc ? qf : qc;
				qmax = qmax > qb ? qmax : qb;
				qmax += dt*gq;

				// QMIN=MIN(QF,QC,QB) +dt*GQ
				qmin = qf < qc ? qf : qc;
				qmin = qmin < qb ? qmin : qb;
				qmin += dt*gq;

				// TMAX=MAX(TF,TC,TB) +dt*GT
				tmax = tf > tc ? tf : tc;
				tmax = tmax > tb ? tmax : tb;
				tmax += dt*gt;

				// TMIN=MIN(TF,TC,TB) +dt*GT
				tmin = tf < tc ? tf : tc;
				tmin = tmin < tb ? tmin : tb;
				tmin += dt*gt;

				// U2MAX=MAX(U2F,U2C,U2B) +dt*GU2
				U2MAX = u2f > u2c ? u2f : u2c;
				U2MAX = U2MAX > u2b ? U2MAX : u2b;
				U2MAX += dt*gu2;

				// U2MIN=MIN(U2F,U2C,U2B) +dt*GU2
				U2MIN = u2f < u2c ? u2f : u2c;
				U2MIN = U2MIN < u2b ? U2MIN : u2b;
				U2MIN += dt*gu2;

				// U1MAX=MAX(U1F,U1C,U1B) +dt*GU1
				u1max = u1f > u1c ? u1f : u1c;
				u1max = u1max > u1b ? u1max : u1b;
				u1max += dt*gu1;

				// U1MIN=MIN(U1F,U1C,U1B) +dt*GU1 
				u1min = u1f < u1c ? u1f : u1c;
				u1min = u1min < u1b ? u1min : u1b;
				u1min += dt*gu1;

				// invariants correction
				if (rfn > rmax) rfn = rmax;
				if (rfn < rmin) rfn = rmin;

				if (qbn > qmax) qbn = qmax;
				if (qbn < qmin) qbn = qmin;

				if (tfn > tmax) tfn = tmax;
				if (tfn < tmin) tfn = tmin;

				if (tbn > tmax) tbn = tmax;
				if (tbn < tmin) tbn = tmin;

				if (u2fn > U2MAX) u2fn = U2MAX;
				if (u2fn < U2MIN) u2fn = U2MIN;

				if (u2bn > U2MAX) u2bn = U2MAX;
				if (u2bn < U2MIN) u2bn = U2MIN;

				if (u1fn > u1max) u1fn = u1max;
				if (u1fn < u1min) u1fn = u1min;

				if (u1bn > u1max) u1bn = u1max;
				if (u1bn < u1min) u1bn = u1min;

				// put invariants to buffers
				rBuf[k + 1] = rfn;
				qBuf[k] = qbn;
				tfBuf[k + 1] = tfn;
				tbBuf[k] = tbn;
				u2fBuf[k + 1] = u1fn;
				u2bBuf[k] = u1bn;
				u3fBuf[k + 1] = u1fn;
				u3bBuf[k] = u1bn;
			}

			// boundary conditions along the X3 axis
			// assignment of boundary invatiants and add them to the buffer arrays

			// periodicity conditions
			rBuf[1] = rBuf[n3];
			tfBuf[1] = tfBuf[n3];
			u2fBuf[1] = u2fBuf[n3];
			u3fBuf[1] = u3fBuf[n3];

			// periodicity conditions
			qBuf[n3] = qBuf[1];
			tbBuf[n3] = tbBuf[1];
			u2bBuf[n3] = u2bBuf[1];
			u3bBuf[n3] = u3bBuf[1];

			// the flow variables calculations
			for (int k = 1; k <= n3; k++)
			{
				ro0b = ronCon[i][j][k - 1];
				ro0f = ronCon[i][j][k];

				rn = rBuf[k];
				qn = qBuf[k];

				pn = (rn - qn)*sound*ro0_g / 2;
				un = (rn + qn) / 2;

				ron = (ro0_g + pn / (sound*sound));

				ucf = u3nCon[i][j][k];
				ucb = u3nCon[i][j][k - 1];

				if (ucf >= 0 && ucb > 0)
				{
					tn = tfBuf[j];
					u2n = u2fBuf[j];
					u1n = u3fBuf[j];
				}
				else if (ucf <= 0 && ucb <= 0)
				{
					tn = tbBuf[j];
					u2n = u2bBuf[j];
					u1n = u3bBuf[j];
				}
				else if (ucb >= 0 && ucf <= 0)
				{
					if (ucb > -ucf)
					{
						tn = tfBuf[j];
						u2n = u2fBuf[j];
						u1n = u3fBuf[j];
					}
					else
					{
						tn = tbBuf[j];
						u2n = u2bBuf[j];
						u1n = u3bBuf[j];
					}
				}
				else
				{
					if (ucb <= 0 && ucf >= 0)
					{
						tn = tnCon[i][j][k] + tnCon[i][j][k - 1] - t3[i][j][k];
						u2n = u2nCon[i][j][k] + u2nCon[i][j][k - 1] - u23[i][j][k];
						u1n = u1nCon[i][j][k] + u1nCon[i][j][k - 1] - u13[i][j][k];
					}
				}

				p3[i][j][k] = pn;
				u33[i][j][k] = un;
				ro3[i][j][k] = ron;
				t3[i][j][k] = tn;
				u23[i][j][k] = u2n;
				u13[i][j][k] = u1n;
			}

			// the flow variable calculations on the bottom border
			p3[i][j][1] = p3[i][j][n3];
			u33[i][j][1] = u33[i][j][n3];
			ro3[i][j][1] = ro3[i][j][n3];
			t3[i][j][1] = t3[i][j][n3];
			u23[i][j][1] = u23[i][j][n3];
			u13[i][j][1] = u13[i][j][n3];

			// inlet conditions
			qn = qBuf[1];
			rn = u3Inlet + (ronCon[i][j][1] - ro0_g)*sound / ro0_g;
			un = (rn + qn) / 2;
			pn = (rn - qn)*sound*ro0_g / 2;
			ron = ro0_g + pn / sound / sound;
			u2n = u2Inlet;
			u1n = u1Inlet;
			tn = tInlet;
			p3[i][j][1] = pn;
			u33[i][j][1] = un;
			ro3[i][j][1] = ron;
			t3[i][j][1] = tn;
			u23[i][j][1] = u2n;
			u13[i][j][1] = u1n;

			// outlet conditions
			rn = rBuf[n3];
			pn = pOutlet;
			un = rn - pn / ro0_g / sound;
			tn = tfBuf[n3];
			u2n = u2fBuf[n3];
			u1n = u3fBuf[n3];
			p3[i][j][n3] = pn;
			u33[i][j][n3] = un;
			ro3[i][j][n3] = ron;
			t3[i][j][n3] = tn;
			u23[i][j][n3] = u2n;
			u13[i][j][n3] = u1n;
		}
	}
}

void StressTensor()
{
	// initialization of friction stress arrays
	for (int i = 0; i <= n1; ++i) {
		for (int j = 0; j <= n2; ++j) {
			for (int k = 0; k <= n3; ++k) {
				sigm11[i][j][k] = 0.0;
				sigm21[i][j][k] = 0.0;
				sigm31[i][j][k] = 0.0;

				sigm12[i][j][k] = 0.0;
				sigm22[i][j][k] = 0.0;
				sigm32[i][j][k] = 0.0;

				sigm13[i][j][k] = 0.0;
				sigm23[i][j][k] = 0.0;
				sigm33[i][j][k] = 0.0;
			}
		}
	}

	// #####################################################
	// 				boundary conditions
	// #####################################################

	// no-slip condition on the boundary faces perpendicular to x1
	x1[0] = x1[1];
	x1[n1+1] = x1[n1];

	for (int j = 1; j < n2 ; ++j) {
		for (int k = 1; k < n3; ++k) {
			u1Con[0][j][k] = 0.;
			u2Con[0][j][k] = 0.;
			u3Con[0][j][k] = 0.;

			u1Con[n1][j][k] = 0.;
			u2Con[n1][j][k] = 0.;
			u3Con[n1][j][k] = 0.;
		}

	}

	// periodic contition on the boundary faces perpendicular to X2
	// nothing to do since we have uniform grid

	// periodic contition on the boundary faces perpendicular to X3
	// nothing to do since we have uniform grid

	// #####################################################
	// 				bypassing along the faces
	// #####################################################

	double xle, xlw, xlt, xln, u1c, u1ce, u2c, u2ce, u3c, u3ce;

	// bypassing along the face perpendicular to x1
	for (int i = 1; i <= n1; ++i) {
		for (int j = 1; j < n2; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[i] - 1);

				// velocity components in cell centers
				u1c = u1Con[i][j][k];
				u1ce = u1Con[i-1][j][k];

				u2c = u2Con[i][j][k];
				u2ce = u2Con[i-1][j][k];

				u3c = u3Con[i][j][k];
				u3ce = u3Con[i-1][j][k];

				// friction stress
				sigm11[i][j][k]=-VIS*xle*(u1ce-u1c)/dx1;
				sigm21[i][j][k]=-VIS*xle*(u2ce-u2c)/dx1;
				sigm31[i][j][k]=-VIS*xle*(u3ce-u3c)/dx1;
			}
		}
	}

	double u1cn, u2cn, u3cn;

	// bypassing along the face perpenditcular to X2
	for (int i = 1; i < n1; ++i) {
		for (int j = 1; j <= n2; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[i] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l-1)*(x1[i-1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);
				// geometric factor of cylindricity
				xln = xlt;

				// velocity components in cell centers
				u1c = u1Con[i][j][k];
				u1cn = u1Con[i][j-1][k];

				u2c = u2Con[i][j][k];
				u2cn = u2Con[i][j-1][k];

				u3c = u3Con[i][j][k];
				u3cn = u3Con[i][j-1][k];

				// friction stress
				sigm12[i][j][k]=-VIS*((u1cn-u1c)/dx2 -(l-1)*(u2c+u2cn))/xln;
				sigm22[i][j][k]=-VIS*((u2cn-u2c)/dx2 +(l-1)*(u1c+u1cn))/xln;
				sigm32[i][j][k]=-VIS*(u3cn-u3c)/dx2;
			}
		}
	}

	double u1ct, u2ct, u3ct;

	// bypassing along the face perpenditcular to X3
	for (int i = 1; i < n1; ++i) {
		for (int j = 1; j < n2; ++j) {
			for (int k = 1; k <= n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[i] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l-1)*(x1[i-1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);

				// velocity components in the cell centers
				u1c = u1Con[i][j][k];
				u1ct = u1Con[i][j][k-1];

				u2c = u2Con[i][j][k];
				u2ct = u2Con[i][j][k-1];

				u3c = u3Con[i][j][k];
				u3ct = u3Con[i][j][k-1];

				// friction stress
				sigm13[i][j][k]=-VIS*xlt*(u1ct-u1c)/dx3;
				sigm23[i][j][k]=-VIS*xlt*(u2ct-u2c)/dx3;
				sigm33[i][j][k]=-VIS*xlt*(u3ct-u3c)/dx3;
			}
		}
	}

	// #####################################################
	// 				friction forces computation
	// #####################################################

	double ds1, ds2, ds3, sigm1c, sigm2c;

	// area of the face perpendicuar to x1
	ds1 = dx2*dx3;
	ds2 = dx1*dx3;
	ds3 = dx1*dx2;

	for (int i = 1; i < n1; ++i) {
		for (int j = 1; j < n2; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[i] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l-1)*(x1[i-1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);

				sigm1c = VIS*u1Con[i][j][k]/xlt;
				sigm2c = VIS*u2Con[i][j][k]/xlt;

				// friction forces
				f1[i][j][k] =
						(sigm11[i+1][j][k] - sigm11[i][j][k]) * ds1 +
						(sigm12[i][j+1][k] - sigm12[i][j][k]) * ds2 +
						(sigm13[i][j][k+1] - sigm13[i][j][k]) * ds3 -
						(l-1)*sigm1c*dx1*dx2*dx3;

				f2[i][j][k] =
						(sigm21[i+1][j][k] - sigm21[i][j][k]) * ds1 +
						(sigm22[i][j+1][k] - sigm22[i][j][k]) * ds2 +
						(sigm23[i][j][k+1] - sigm23[i][j][k]) * ds3 -
						(l-1)*sigm2c*dx1*dx2*dx3;

				f3[i][j][k] =
						(sigm21[i+1][j][k] - sigm21[i][j][k]) * ds1 +
						(sigm22[i][j+1][k] - sigm22[i][j][k]) * ds2 +
						(sigm23[i][j][k+1] - sigm23[i][j][k]) * ds3;
			}
		}
	}

}

void UseForces()
{
	double xle, xlw, dvc, roc, rocn;

	for (int i = 1; i < n1; ++i) {
		for (int j = 1; j < n2; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric factor of cylindricity
				xle = 1+(l-1)*(x1[i+1]-1);
				xlw = 1+(l-1)*(x1[i]-1);
				// cell volume
				dvc = 0.5*(xle+xlw)*dx1*dx2*dx3;

				roc = roCon[i][j][k];
				rocn = ronCon[i][j][k];

				u1nCon[i][j][k] = (roc*dvc*u1nCon[i][j][k] + 0.5*dt*f1[i][j][k])/(dvc*rocn);
				u2nCon[i][j][k] = (roc*dvc*u2nCon[i][j][k] + 0.5*dt*f2[i][j][k])/(dvc*rocn);
				u3nCon[i][j][k] = (roc*dvc*u3nCon[i][j][k] + 0.5*dt*f3[i][j][k])/(dvc*rocn);
			}
		}
	}

}

void WriteData()
{
	char filename[50];

	sprintf_s(filename, 50, "out_%d.tec", nStep);

	FILE *fd;
	fopen_s(&fd, filename, "w");

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

void WriteDataParaView()
{
	char filename[50];

	sprintf_s(filename, 50, "out_%d.vtk", nStep);

	FILE *fd;
	fopen_s(&fd, filename, "w");

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
		for (int i = 1; i < n1; i++)
		{
			for (int j = 1; j < n2; j++)
			{
				fprintf(fd, "%f ", u1nCon[i][j][k]);
			}
		}
	}
	
	fprintf(fd, "\nscalars U2 float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n1; i++)
		{
			for (int j = 1; j < n2; j++)
			{
				fprintf(fd, "%f ", u2nCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars U3 float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n1; i++)
		{
			for (int j = 1; j < n2; j++)
			{
				fprintf(fd, "%f ", u3nCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars PC float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n1; i++)
		{
			for (int j = 1; j < n2; j++)
			{
				fprintf(fd, "%f ", ronCon[i][j][k]);
			}
		}
	}

	fprintf(fd, "\nscalars TC float\nLOOKUP_TABLE default\n");
	for (int k = 1; k < n3; k++)
	{
		for (int i = 1; i < n1; i++)
		{
			for (int j = 1; j < n2; j++)
			{
				fprintf(fd, "%f ", tnCon[i][j][k]);
			}
		}
	}
	
	fclose(fd);
}

void FreeMemory()
{
	free(x1);
	free(x2);
	free(x3);

	deallocate3D(roCon, n1, n2);
	deallocate3D(u1Con, n1, n2);
	deallocate3D(u2Con, n1, n2);
	deallocate3D(u3Con, n1, n2);
	deallocate3D(tCon, n1, n2);

	deallocate3D(ronCon, n1, n2);
	deallocate3D(u1nCon, n1, n2);
	deallocate3D(u2nCon, n1, n2);
	deallocate3D(u3nCon, n1, n2);
	deallocate3D(tnCon, n1, n2);

	deallocate3D(ro1, n1+1, n2);
	deallocate3D(t1, n1+1, n2);
	deallocate3D(u11, n1+1, n2);
	deallocate3D(u21, n1+1, n2);
	deallocate3D(u31, n1+1, n2);
	deallocate3D(p1, n1+1, n2);

	deallocate3D(ro2, n1, n2+1);
	deallocate3D(t2, n1, n2+1);
	deallocate3D(u12, n1, n2+1);
	deallocate3D(u22, n1, n2+1);
	deallocate3D(u32, n1, n2+1);
	deallocate3D(p2, n1, n2+1);

	deallocate3D(ro3, n1, n2);
	deallocate3D(t3, n1, n2);
	deallocate3D(u13, n1, n2);
	deallocate3D(u23, n1, n2);
	deallocate3D(u33, n1, n2);
	deallocate3D(p3, n1, n2);

	deallocate3D(f1, n1, n2);
	deallocate3D(f2, n1, n2);
	deallocate3D(f3, n1, n2);

	free(rBuf);
	free(qBuf);
	free(tfBuf);
	free(tbBuf);
	free(u2fBuf);
	free(u2bBuf);
	free(u3fBuf);
	free(u3bBuf);

	deallocate3D(sigm11, n1, n2);
	deallocate3D(sigm21, n1, n2);
	deallocate3D(sigm31, n1, n2);

	deallocate3D(sigm12, n1, n2);
	deallocate3D(sigm22, n1, n2);
	deallocate3D(sigm32, n1, n2);

	deallocate3D(sigm13, n1, n2);
	deallocate3D(sigm23, n1, n2);
	deallocate3D(sigm33, n1, n2);
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