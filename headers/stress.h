#pragma once
#ifndef _STRESS_H
#define _STRESS_H

//	--------------------------------------------------------------------------------------
//	SIGMA11(:,:,:) - friction stress in the direction X1 to the faces perpendicular to the axis X1
//	SIGMA21(:,:,:) - friction stress in the direction X2 to the faces perpendicular to the axis X1
//	SIGMA31(:,:,:) - friction stress in the direction X3 to the faces perpendicular to the axis X1
//	SIGMA12(:,:,:) - friction stress in the direction X1 to the faces perpendicular to the axis X2
//	SIGMA22(:,:,:) - friction stress in the direction X2 to the faces perpendicular to the axis X2
//	SIGMA32(:,:,:) - friction stress in the direction X3 to the faces perpendicular to the axis X2
//	SIGMA13(:,:,:) - friction stress in the direction X1 to the faces perpendicular to the axis X3
//	SIGMA23(:,:,:) - friction stress in the direction X2 to the faces perpendicular to the axis X3
//	SIGMA33(:,:,:) - friction stress in the direction X3 to the faces perpendicular to the axis X3

double  ***SIGM11, ***SIGM21, ***SIGM31,
        ***SIGM12, ***SIGM22, ***SIGM32,
        ***SIGM13, ***SIGM23, ***SIGM33;

#endif
