#pragma once
#ifndef _STRESS_H
#define _STRESS_H

//     --------------------------------------------------------------------------------------
//     SIGMA11(:,:,:) - shear stress in the direction X1 to the faces perpendicular to the axis X1
//     SIGMA21(:,:,:) - shear stress in the direction X2 to the faces perpendicular to the axis X1
//     SIGMA31(:,:,:) - shear stress in the direction X3 to the faces perpendicular to the axis X1
//     SIGMA12(:,:,:) - shear stress in the direction X1 to the faces perpendicular to the axis X2
//     SIGMA22(:,:,:) - shear stress in the direction X2 to the faces perpendicular to the axis X2
//     SIGMA32(:,:,:) - shear stress in the direction X3 to the faces perpendicular to the axis X2
//     SIGMA13(:,:,:) - shear stress in the direction X1 to the faces perpendicular to the axis X3
//     SIGMA23(:,:,:) - shear stress in the direction X2 to the faces perpendicular to the axis X3
//     SIGMA33(:,:,:) - shear stress in the direction X3 to the faces perpendicular to the axis X3

double  ***SIGM11, ***SIGM21, ***SIGM31,
        ***SIGM12, ***SIGM22, ***SIGM32,
        ***SIGM13, ***SIGM23, ***SIGM33;

#endif
