#include "stress.h"

//	---------------------------------------------------------------------------------------
//	f1(:,:,:) - force (friction, etc.) along the axis x1
//	f2(:,:,:) - force (friction, etc.) along the axis X2
//	f3(:,:,:) - force (friction, etc.) along the axis X3

double ***f1, ***f2, ***f3;

//	--------------------------------------------------------------------------------------
//	SIGMA11(:,:,:) - friction stress in the direction x1 to the faces perpendicular to the axis x1
//	SIGMA21(:,:,:) - friction stress in the direction X2 to the faces perpendicular to the axis x1
//	SIGMA31(:,:,:) - friction stress in the direction X3 to the faces perpendicular to the axis x1
//	SIGMA12(:,:,:) - friction stress in the direction x1 to the faces perpendicular to the axis X2
//	SIGMA22(:,:,:) - friction stress in the direction X2 to the faces perpendicular to the axis X2
//	SIGMA32(:,:,:) - friction stress in the direction X3 to the faces perpendicular to the axis X2
//	SIGMA13(:,:,:) - friction stress in the direction x1 to the faces perpendicular to the axis X3
//	SIGMA23(:,:,:) - friction stress in the direction X2 to the faces perpendicular to the axis X3
//	SIGMA33(:,:,:) - friction stress in the direction X3 to the faces perpendicular to the axis X3

double  ***sigm11, ***sigm21, ***sigm31,
        ***sigm12, ***sigm22, ***sigm32,
        ***sigm13, ***sigm23, ***sigm33;

void allocateForces(int n1, int n2, int n3)
{
	f1 = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	f2 = allocate3D(n2 + 1, n1 + 1, n3 + 1);
	f3 = allocate3D(n2 + 1, n1 + 1, n3 + 1);
}

void deallocateForces(int n1, int n2, int n3)
{
	deallocate3D(f1, n2 + 1, n1 + 1);
	deallocate3D(f2, n2 + 1, n1 + 1);
	deallocate3D(f3, n2 + 1, n1 + 1);
}


void allocateStress(int n1, int n2, int n3)
{
	sigm11 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	sigm21 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	sigm31 = allocate3D(n2 + 2, n1 + 2, n3 + 2);

	sigm12 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	sigm22 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	sigm32 = allocate3D(n2 + 2, n1 + 2, n3 + 2);

	sigm13 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	sigm23 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
	sigm33 = allocate3D(n2 + 2, n1 + 2, n3 + 2);
}

void deallocateStress(int n1, int n2, int n3)
{
	deallocate3D(sigm11, n2 + 2, n1 + 2);
	deallocate3D(sigm21, n2 + 2, n1 + 2);
	deallocate3D(sigm31, n2 + 2, n1 + 2);

	deallocate3D(sigm12, n2 + 2, n1 + 2);
	deallocate3D(sigm22, n2 + 2, n1 + 2);
	deallocate3D(sigm32, n2 + 2, n1 + 2);

	deallocate3D(sigm13, n2 + 2, n1 + 2);
	deallocate3D(sigm23, n2 + 2, n1 + 2);
	deallocate3D(sigm33, n2 + 2, n1 + 2);
}


// input: n1, n2, n3, l, x1, u1Con, u2Con, u3Con, VIS, dx1, dx2, dx3
// output: sigm11, sigm21, sigm31, sigm12, sigm22, sigm32, sigm13, sigm23, sigm33
// output: f1, f2, f3
void StressTensor()
{
	// initialization of friction stress arrays
	for (int i = 0; i < n2 + 2; ++i) {
		for (int j = 0; j < n1 + 2; ++j) {
			for (int k = 0; k < n3 + 2; ++k) {
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
	// 				bypassing along the faces
	// #####################################################

	double xle, xlw, xlt, xln, u1_c, u1_ce, u2_c, u2_ce, u3_c, u3_ce;

	// bypassing along the face perpendicular to x1
	for (int i = 1; i <= n2; ++i) {
		for (int j = 1; j <= n1; ++j) {
			for (int k = 1; k <= n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[i] - 1);

				// velocity components in cell centers
				u1_c = u1Con[i][j][k];
				u1_ce = u1Con[i-1][j][k];

				u2_c = u2Con[i][j][k];
				u2_ce = u2Con[i-1][j][k];

				u3_c = u3Con[i][j][k];
				u3_ce = u3Con[i-1][j][k];

				// friction stress
				sigm11[i][j][k]=-VIS*xle*(u1_ce-u1_c)/dx1;
				sigm21[i][j][k]=-VIS*xle*(u2_ce-u2_c)/dx1;
				sigm31[i][j][k]=-VIS*xle*(u3_ce-u3_c)/dx1;
			}
		}
	}

	double u1_cn, u2_cn, u3_cn;

	// bypassing along the face perpenditcular to X2
	for (int i = 1; i <= n2; ++i) {
		for (int j = 1; j <= n1; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[j] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l-1)*(x1[j-1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);
				// geometric factor of cylindricity
				xln = xlt;

				// velocity components in cell centers
				u1_c = u1Con[i][j][k];
				u1_cn = u1Con[i][j-1][k];

				u2_c = u2Con[i][j][k];
				u2_cn = u2Con[i][j-1][k];

				u3_c = u3Con[i][j][k];
				u3_cn = u3Con[i][j-1][k];

				// friction stress
				sigm12[i][j][k]=-VIS*((u1_cn-u1_c)/dx2 -(l-1)*(u2_c+u2_cn))/xln;
				sigm22[i][j][k]=-VIS*((u2_cn-u2_c)/dx2 +(l-1)*(u1_c+u1_cn))/xln;
				sigm32[i][j][k]=-VIS*(u3_cn-u3_c)/dx2;
			}
		}
	}

	double u1_ct, u2_ct, u3_ct;

	// bypassing along the face perpenditcular to X3
	for (int i = 1; i <= n2; ++i) {
		for (int j = 1; j <= n1; ++j) {
			for (int k = 1; k <= n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[j] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l-1)*(x1[j-1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);

				// velocity components in the cell centers
				u1_c = u1Con[i][j][k];
				u1_ct = u1Con[i][j][k-1];

				u2_c = u2Con[i][j][k];
				u2_ct = u2Con[i][j][k-1];

				u3_c = u3Con[i][j][k];
				u3_ct = u3Con[i][j][k-1];

				// friction stress
				sigm13[i][j][k]=-VIS*xlt*(u1_ct-u1_c)/dx3;
				sigm23[i][j][k]=-VIS*xlt*(u2_ct-u2_c)/dx3;
				sigm33[i][j][k]=-VIS*xlt*(u3_ct-u3_c)/dx3;
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

	for (int i = 1; i < n2; ++i) {
		for (int j = 1; j < n1; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric characteristics of the computational cell
				// geometric factor of cylindricity
				xle = 1 + (l-1)*(x1[j] - 1);
				// geometric factor of cylindricity
				xlw = 1 + (l-1)*(x1[j-1] - 1);
				// geometric factor of cylindricity
				xlt = 0.5*(xle + xlw);

				sigm1c = VIS*u1Con[i][j][k]/xlt;
				sigm2c = VIS*u2Con[i][j][k]/xlt;

				// friction forces
				f1[i][j][k] =
					(sigm11[i][j + 1][k] - sigm11[i][j][k]) * ds1 +
					(sigm12[i + 1][j][k] - sigm12[i][j][k]) * ds2 +
					(sigm13[i][j][k + 1] - sigm13[i][j][k]) * ds3 -
					(l - 1)*sigm1c*dx1*dx2*dx3;

				f2[i][j][k] =
					(sigm21[i][j + 1][k] - sigm21[i][j][k]) * ds1 +
					(sigm22[i + 1][j][k] - sigm22[i][j][k]) * ds2 +
					(sigm23[i][j][k + 1] - sigm23[i][j][k]) * ds3 -
					(l - 1)*sigm2c*dx1*dx2*dx3;

				f3[i][j][k] =
					(sigm31[i][j + 1][k] - sigm31[i][j][k]) * ds1 +
					(sigm32[i + 1][j][k] - sigm32[i][j][k]) * ds2 +
					(sigm33[i][j][k + 1] - sigm33[i][j][k]) * ds3;
			}
		}
	}
}

// input: l, n1, n2, n3, x1, dx1, dx2, dx3, roCon, ronCon, dt
// output: u1nCon, u2nCon, u3nCon
void UseForces()
{
	double xle, xlw, dvc, ro_c, ro_cn;

	for (int i = 1; i < n2; ++i) {
		for (int j = 1; j < n1; ++j) {
			for (int k = 1; k < n3; ++k) {
				// geometric factor of cylindricity
				xle = 1+(l-1)*(x1[j+1]-1);
				xlw = 1+(l-1)*(x1[j]-1);
				// cell volume
				dvc = 0.5*(xle+xlw)*dx1*dx2*dx3;

				ro_c = roCon[i][j][k];
				ro_cn = ronCon[i][j][k];

				u1nCon[i][j][k] = (ro_c*dvc*u1nCon[i][j][k] + 0.5*dt*f1[i][j][k])/(dvc*ro_cn);
				u2nCon[i][j][k] = (ro_c*dvc*u2nCon[i][j][k] + 0.5*dt*f2[i][j][k])/(dvc*ro_cn);
				u3nCon[i][j][k] = (ro_c*dvc*u3nCon[i][j][k] + 0.5*dt*f3[i][j][k])/(dvc*ro_cn);
			}
		}
	}
}
