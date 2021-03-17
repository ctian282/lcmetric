cdef extern from "c++/utils.h":

     void _interp[T](T *d, double *dx, int nx, int ny, int nz,
             T *x_list, int ns, T *res) except +
