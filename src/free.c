#include "free.h"

void FreeMemory()
{
	free(x1);
	free(x2);
	free(x3);

	deallocate3D(roCon, n2 + 1, n1 + 1);
	deallocate3D(u1Con, n2 + 1, n1 + 1);
	deallocate3D(u2Con, n2 + 1, n1 + 1);
	deallocate3D(u3Con, n2 + 1, n1 + 1);
	deallocate3D(tCon, n2 + 1, n1 + 1);

	deallocate3D(ronCon, n2 + 1, n1 + 1);
	deallocate3D(u1nCon, n2 + 1, n1 + 1);
	deallocate3D(u2nCon, n2 + 1, n1 + 1);
	deallocate3D(u3nCon, n2 + 1, n1 + 1);
	deallocate3D(tnCon, n2 + 1, n1 + 1);

	deallocate3D(ro1, n2 + 2, n1 + 2);
	deallocate3D(t1, n2 + 2, n1 + 2);
	deallocate3D(u11, n2 + 2, n1 + 2);
	deallocate3D(u21, n2 + 2, n1 + 2);
	deallocate3D(u31, n2 + 2, n1 + 2);
	deallocate3D(p1, n2 + 2, n1 + 2);

	deallocate3D(ro2, n2 + 2, n1 + 2);
	deallocate3D(t2, n2 + 2, n1 + 2);
	deallocate3D(u12, n2 + 2, n1 + 2);
	deallocate3D(u22, n2 + 2, n1 + 2);
	deallocate3D(u32, n2 + 2, n1 + 2);
	deallocate3D(p2, n2 + 2, n1 + 2);

	deallocate3D(ro3, n2 + 2, n1 + 2);
	deallocate3D(t3, n2 + 2, n1 + 2);
	deallocate3D(u13, n2 + 2, n1 + 2);
	deallocate3D(u23, n2 + 2, n1 + 2);
	deallocate3D(u33, n2 + 2, n1 + 2);
	deallocate3D(p3, n2 + 2, n1 + 2);

	deallocateForces(n1, n2, n3);
	deallocateStress(n1, n2, n3);

	free(rBuf);
	free(qBuf);
	free(tfBuf);
	free(tbBuf);
	free(u2fBuf);
	free(u2bBuf);
	free(u3fBuf);
	free(u3bBuf);
}

void deallocate3D(double*** arr, int n1, int n2)
{
	for (int i = 0; i < n1; ++i) {
		for (int j = 0; j < n2; ++j) {
			free(arr[i][j]);
		}
		free(arr[i]);
	}
	free(arr);
}
