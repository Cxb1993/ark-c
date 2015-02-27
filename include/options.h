#ifndef _OPTIONS_H
#define _OPTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include "getopt.h"


typedef struct _Options {
	int mode;
    const char *program_name;
    const char *version_name, *version_number;
	int debug_mode;

	int index_geometry, n1g, n2g, n3g, nPrint, nStop;
	double delta, kappa, cfl;

	char *input_file;
    char *output_file;
    int gpu_mode;
} Options;

Options parseOptions(int argc, char *argv[]);

void helpPrint(Options opt);
void versionPrint(Options opt);
void errorPrint(Options opt);

void infoPrint(Options opt);

#endif /* _OPTIONS_H */
