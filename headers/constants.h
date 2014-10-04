#pragma once
#ifndef _CONSTANTS_H
#define _CONSTANTS_H

//     -----------------------------------------------------------------------------------
//      SOUND -	sound speed
//      RO0G -	unperturbed density of the liquid
//      RO0S -	unperturbed density of the barriers material
//      DT -	time step
//      VIS -	kinematic viscosity
//      U10 -	initial speed along the axis X1
//      U20 -	initial speed along the axis X2
//      U30 -	initial speed along the axis X3
//      T0 -	initial temperature
//      TIME -	current time
//      CFL -	Courant number

double SOUND, RO0G, RO0S, DT, VIS, U10, U20, U30, TIME, CFL;

const double PI = 3.141592653589793239;

#endif

