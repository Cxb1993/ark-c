#pragma once
#ifndef _BULK_H
#define _BULK_H

//	---------------------------------------------------------------------------------
//	L - geometry index: L=2 - cylindrical coordinates, L=1 - cartesian coordinates
//	N1 - number of grid nodes in X1 direction for one processor
//	N2 - number of grid nodes in X2 direction for one processor
//	N3 - number of grid nodes in X3 direction for one processor
//	NSTOP - total number of time steps
//	NPRINT - print interval

int L, N1, N2, N3, NSTOP, NPRINT, NSTEP, N1G, N2G, N3G;

//	---------------------------------------------------------------------------------
//	X1E - coordinates of the east border
//	X1W - coordinates of the west border
//	X2N - coordinates of the north border
//	X2S - coordinates of the south border
//	X3T - coordinates of the top border
//	X3B - coordinates of the bottom border

double X1E, X1W, X2N, X2S, X3T, X3B;

//	---------------------------------------------------------------------------------
//	DX1 - grid step along X1 axis
//	DX2 - grid step along X2 axis
//	DX3 - grid step along X3 axis

double DX1, DX2, DX3;

//	---------------------------------------------------------------------------------
//	X1(:) - coordinates of the grid nodes along the axis X1
//	X2(:) - coordinates of the grid nodes along the axis X2
//	X3(:) - coordinates of the grid nodes along the axis X3

double *X1, *X2, *X3;

//	----------------------------------------------------------------------------------
//	ROCON(:,:,:) - conservative density at the current time step
//	U1CON(:,:,:) - conservative speed at the current time step along the axis X1
//	U2CON(:,:,:) - conservative speed at the current time step along the axis X2
//	U3CON(:,:,:) - conservative speed at the current time step along the axis X3
//	TCON(:,:,:) - conservative temperature at the current time step

double ***ROCON, ***U1CON, ***U2CON, ***U3CON, ***TCON;

//	----------------------------------------------------------------------------------
//	RONCON(:,:,:) - conservative density on the next time step
//	U1NCON(:,:,:) - conservative speed on the next time step along the axis X1
//	U2NCON(:,:,:) - conservative speed on the next time step along the axis X2
//	U3NCON(:,:,:) - conservative speed on the next time step along the axis X3
//	TNCON(:,:,:) - conservative temperature at the next time step

double ***RONCON, ***U1NCON, ***U2NCON, ***U3NCON, ***TNCON;

//	----------------------------------------------------------------------------------
//	P1(:,:,:) -  pressure on the faces perpendicular to the axis X1
//	RO1(:,:,:) - density on the faces perpendicular to the axis X1
//	U11(:,:,:) - speed along the axis X1 on the faces perpendicular to the axis X1
//	U21(:,:,:) - speed along the axis X2 on the faces perpendicular to the axis X1
//	U31(:,:,:) - speed along the axis X3 on the faces perpendicular to the axis X1
//	T1(:,:,:) -  temperature on the faces perpendicular to the axis X1

double ***P1, ***RO1, ***U11, ***U21, ***U31, ***T1;

//	----------------------------------------------------------------------------------
//	P2(:,:,:) -  pressure on the faces perpendicular to the axis X2
//	RO2(:,:,:) - density on the faces perpendicular to the axis X2
//	U12(:,:,:) - speed along the axis X1 on the faces perpendicular to the axis X2
//	U22(:,:,:) - speed along the axis X2 on the faces perpendicular to the axis X2
//	U32(:,:,:) - speed along the axis X3 on the faces perpendicular to the axis X2
//	T2(:,:,:) -  temperature on the faces perpendicular to the axis X2

double ***P2, ***RO2, ***U12, ***U22, ***U32, ***T2;

//	----------------------------------------------------------------------------------
//	P3(:,:,:) -  pressure on the faces perpendicular to the axis X3
//	RO3(:,:,:) - density on the faces perpendicular to the axis X3
//	U13(:,:,:) - speed along the axis X1 on the faces perpendicular to the axis X3
//	U23(:,:,:) - speed along the axis X2 on the faces perpendicular to the axis X3
//	U33(:,:,:) - speed along the axis X3 on the faces perpendicular to the axis X3
//	T3(:,:,:) -  temperature on the faces perpendicular to the axis X3

double ***P3, ***RO3, ***U13, ***U23, ***U33, ***T3;

//	-----------------------------------------------------------------------------------
//	R(:)- one-dimensional buffer array for storing Riemann R invariant calculated values
//	Q(:)- one-dimensional buffer array for storing Riemann Q invariant calculated values

double *RBUF, *QBUF, *TFBUF, *TBBUF, *U2FBUF, *U2BBUF, *U3FBUF, *U3BBUF;

#endif