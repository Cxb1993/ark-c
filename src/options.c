#include "options.h"

Options init();
Options parse(int argc, char *argv[]);
Options checkOptions(Options opt);

Options parseOptions(int argc, char *argv[])
{
	Options opt = parse(argc, argv);
	if (!opt.mode)
		return opt;
    return checkOptions(opt);
}

Options init()
{
	Options opt;
	opt.mode = -1;
    opt.program_name = opt.version_name = opt.version_number = 0;
	opt.debug_mode = 0;

	opt.index_geometry = 0;
	opt.n1g = opt.n2g = opt.n3g = opt.nPrint = opt.nStop = 0;
	opt.delta = opt.kappa = opt.cfl = 0;

    opt.input_file = opt.output_file = 0;
    opt.gpu_mode = 0;
	return opt;
}

Options parse(int argc, char *argv[])
{
	Options opt = init();
#ifdef PROGRAM_NAME
    opt.program_name = PROGRAM_NAME;
#else
    opt.program_name = argv[0];
#endif
#ifdef VERSION
    opt.version_name = VERSION;
#else
    opt.version_name = "UNKNOW";
#endif
#ifdef VERSION_NUMBER
    opt.version_number = VERSION_NUMBER;
#else
    opt.version_number = "UNKNOW";
#endif
    struct option longopts[] = {
        {"help",                    no_argument,             NULL, 'h'},
        {"version",                 no_argument,             NULL, 'v'},
        {"debug",                   required_argument,       NULL, 'g'},
        {"index-geometry",          required_argument,       NULL, 'l'},
        {"number-node-x1",          required_argument,       NULL, 'x'},
        {"number-node-x2",          required_argument,       NULL, 'y'},
        {"number-node-x3",          required_argument,       NULL, 'z'},
        {"interval-print",          required_argument,       NULL, 'p'},
        {"number-steps",            required_argument,       NULL, 's'},
        {"delta",                   required_argument,       NULL, 'd'},
        {"kappa",                   required_argument,       NULL, 'k'},
        {"cfl",                     required_argument,       NULL, 'c'},
        {"input",                   required_argument,       NULL, 'f'},
        {"output",                  required_argument,       NULL, 'o'},
        {"gpu",                     no_argument,             NULL, 'u'},
        {0, 0, 0, 0}
    };
    int oc;
    int longindex = -1;
    const char *optstring = ":hvg:l:x:y:z:p:s:d:k:c:f:o:u"; // opterr = 0, because ":..."
    while ((oc = getopt_long(argc, argv, optstring, longopts, &longindex)) != -1) {
        switch (oc) {
        case 'h':
            opt.mode = 1;
            break;
        case 'v':
            opt.mode = 2;
            break;
        case 'g':
            opt.debug_mode = atoi(optarg);
            break;
        case 'l':
            opt.index_geometry = atoi(optarg);
            break;
        case 'x':
            opt.n1g = atoi(optarg);
            break;
        case 'y':
            opt.n2g = atoi(optarg);
            break;
        case 'z':
            opt.n3g = atoi(optarg);
            break;
        case 'p':
            opt.nPrint = atoi(optarg);
            break;
        case 's':
            opt.nStop = atoi(optarg);
            break;
        case 'd':
            opt.delta = atof(optarg);
            break;
        case 'k':
            opt.kappa = atof(optarg);
            break;
        case 'c':
            opt.cfl = atof(optarg);
            break;
        case 'f':
            opt.input_file = optarg;
            break;
        case 'o':
            opt.output_file = optarg;
            break;
        case 'u':
            opt.gpu_mode = 1;
            break;
        case 0: // nothing do
            break;
        case ':':
            opt.mode = -1; // TODO: error
            break;
        case '?':
        default:
            opt.mode = -1; // TODO: error
            break;
        }
        longindex = -1;
    }
    //if (optind != argc - 1)
        //error_mode = true; // TODO: error
	return opt;
}

Options checkOptions(Options opt)
{
	return opt;
}

void helpPrint(Options opt)
{
    printf("Usage: %s OPTION...\n \
		\n \
		This option must be present:\n \
		-l, --index-geometry   geometry index: L=1 - coordinates, L=2 - cylindrical coordinates\n \
		-x, --number-node-x1   number of nodes of the computational grid in the direction X1\n \
		-y, --number-node-x2   number of nodes of the computational grid in the direction X2\n \
		-z, --number-node-x3   number of nodes of the computational grid in the direction X3\n \
		-p, --interval-print   interval of print\n \
		-s, --number-steps     complete number of steps\n \
		-d, --delta\n \
		-k, --kappa            coefficient of Kappa\n \
		-c, --cfl              number of Courant\n \
		\n \
		Additional options:\n \
		-h, --help             display this help and exit\n \
		-v, --version          ouput version information and exit\n \
		\n \
		Examples:\n \
		kovcheg -l 2 -x 32 -y 32 -z 64 -p 50 -s 1000 -d 1.0 -k 1.0 -c 0.3\n \
		kovcheg -l 2 -x 64 -y 64 -z 64 --interval-print 100 -s 500 --delta 1.0 -k 1 --cfl 0.3\n",
		opt.program_name);
}

void versionPrint(Options opt)
{
    printf("%s (%s) %s\n", opt.program_name, opt.version_name, opt.version_number);
}

void errorPrint(Options opt)
{
    printf("Error\n");
}

void infoPrint(Options opt)
{
	printf(" \
		mode = %d\n \
		program_name = %s\n \
		version_name = %s\n \
		version_number = %s\n \
		debug_mode = %d\n \
		index_geometry = %d\n \
		n1g = %d\n \
		n2g = %d\n \
		n3g = %d\n \
		nPrint = %d\n \
		nStop = %d\n \
		delta = %f\n \
		kappa = %f\n \
		cfl = %f\n \
		input_file = %s\n \
		output_file = %s\n \
		gpu_mode = %d\n",
		opt.mode, opt.program_name, opt.version_name, opt.version_number, opt.debug_mode,
		opt.index_geometry, opt.n1g, opt.n2g, opt.n3g, opt.nPrint, opt.nStop,
		opt.delta, opt.kappa, opt.cfl,
		opt.input_file, opt.output_file, opt.gpu_mode);
}
