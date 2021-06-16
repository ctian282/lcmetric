
import numpy as npy
import healpy as hp

from libcpp cimport bool

cimport cython
cimport numpy as npy
from cython.parallel import prange

cimport typedefs
from typedefs cimport comp_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef relax_Phi(int npix, comp_t[::1] Phi_p1, \
               comp_t [::1] Phi_m1, comp_t [::1] Pi, comp_t[::1] delta,\
               comp_t[::1] vw, comp_t[::1] rhs, \
               comp_t[::1] res, long long[::1] lm, double tau, double a, \
               double Hubble, double Hubble_0, \
               double Hubble_dt, double Omega_m, double dtau):
    cdef Py_ssize_t p
    for p in prange(npix, nogil=True):
    #for p in range(npix):
        res[p] = ( ( ((-2 / tau + 2 * Hubble) \
                      * (Pi[p] +  (Phi_p1[p] - Phi_m1[p]) / (2 * dtau) )\
                     + 1.5 * Hubble_0**2 * Omega_m / a * delta[p]\
                     + 3 * Hubble_0**2 * Omega_m / a * vw[p]\
                    ) + rhs[p])\
                  - (Phi_p1[p] + Phi_m1[p]) / dtau**2) \
                  / (-2 / dtau**2 + lm[p] / tau**2 \
                     + 2.0 * \
                     (Hubble_dt - Hubble**2 \
                      - 1.5 ** 2 * Hubble_0**2 * Omega_m / a))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dPi_dt(int npix, comp_t [::1] Phi, comp_t[::1] Phi_p1, \
            comp_t [::1] Phi_m1,
            comp_t [::1] Pi, comp_t[::1] vw,\
            comp_t[::1] res, double a, \
            double Hubble, double Hubble_0, \
            double Hubble_dt, double Omega_m, double dtau):
        cdef Py_ssize_t p
        for p in prange(npix, nogil=True):
        #for p in range(npix):
            # res[p] = -( (Phi_p1[p] + Phi_m1[p] - 2 * Phi[p]) / dtau**2 \
            #            ) - 3 * Hubble * \
            #            ( - (Phi_p1[p] - Phi_m1[p]) / (2*dtau) \
            #             ) - 2 * Hubble * Pi[p] \
            #             - (2 * Hubble_dt + Hubble**2) * Phi[p] \
            #             + 1.5 * Hubble_0**2 * Omega_m / a * vw[p]
            res[p] =  - 2 * Hubble * Pi[p] \
                - Hubble *  ( - (Phi_p1[p] - Phi_m1[p]) / (2*dtau))\
                - (2 * Hubble_dt + Hubble**2) * Phi[p] \
                + 1.5 * Hubble_0**2 * Omega_m / a * vw[p]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dPi_dt_upper_bd(int npix, comp_t [::1] Phi,
            comp_t [::1] Phi_m1,
            comp_t [::1] Phi_m2, \
            comp_t [::1] Pi, comp_t[::1] vw,\
            comp_t[::1] res, comp_t [::1] Pi_i, double a,
            double Hubble, double Hubble_0, \
            double Hubble_dt, double Omega_m, double dtau):
        cdef Py_ssize_t p
        for p in prange(npix, nogil=True):
        #for p in range(npix):
            # res[p] = - 3 * Hubble * Pi[p]\
            #     + Hubble * Pi_i[p] \
            #     - (2 * Hubble_dt + Hubble**2) * Phi[p] \
            #     + 1.5 * Hubble_0**2 * Omega_m / a * vw[p]
            res[p] = - 2 * Hubble * Pi[p]\
                - Hubble * ( (Phi_m2[p] - Phi[p])/ (2 * dtau) ) \
                - (2 * Hubble_dt + Hubble**2) * Phi[p] \
                + 1.5 * Hubble_0**2 * Omega_m / a * vw[p]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dPi_dt_lower_bd(int npix, comp_t [::1] Phi,
                     comp_t [::1] Phi_p1,
                     comp_t [::1] Phi_p2, \
                     comp_t [::1] Pi, comp_t[::1] vw,\
                     comp_t[::1] res, double a, \
                     double Hubble, double Hubble_0, \
                     double Hubble_dt, double Omega_m, double dtau):
        cdef Py_ssize_t p
        for p in prange(npix, nogil=True):
        #for p in range(npix):
            res[p] =  - 2 * Hubble * Pi[p] \
                - Hubble *  (  (Phi[p] - Phi_p2[p]) / (2 * dtau))\
                - (2 * Hubble_dt + Hubble**2) * Phi[p] \
                + 1.5 * Hubble_0**2 * Omega_m / a * vw[p]


class Metric:


    def  __init__(self,  nlat, lmax = None, \
                  epsilon = 1e-6, grid = 'healpy', alm_iter = 30, \
                  depth = 3, n_vcycles = 10, npre = 10, npost = 10, verbose = True):
        """Initialize light-cone class
        Parameters
        ----------
        nlat : int
        Is NSIDE when using healpix

        lmax: int, optional
        By default is 2*NSIDE-1

        epsilon: float, has NOT been integrated to multigrid yet
        Leading filename with path of the cone file

        grid: str, optional
        grid type, default is healpy. Currently keep pysh grid support, considering completely
        remove it in the future

        alm_iter: int, optional
        Number of iterations when using healpy map2alm function

        depth: int, optional
        Multigrid V-cycle depth, usually 4-6

        n_vcycles: int, optional
        Number of vcycles

        npre: int, optional
        Number of pre-relaxation at each level in the v-cycle

        npost: int, optional
        Number of post-relaxation at each level in the v-cycle

        verbose: bool, optional
        If output error info for every step, can be expensive!
        Returns
        -------
        void
        """
        self.nside = nlat
        self.Npix = 12* self.nside * self.nside
        self.lmax = lmax
        self.grid = grid
        self.alm_iter = alm_iter
        if(self.lmax == None):
            self.lmax = self.nside * 2 - 1
        self.lm = hp.Alm.getlm(
            self.lmax, npy.array(range( int(1+self.lmax + 0.5 * self.lmax*(1+self.lmax)) )))
        self.lm = -self.lm[0] *(self.lm[0] + 1)
        self.Nalms = hp.Alm.getsize(self.lmax)

        self.depth = depth
        self.n_vcycles = n_vcycles
        self.npre = npre
        self.npost = npost


        self.epsilon = epsilon

        self.verbose = verbose

        # does not really use rho_bar, replaced by H
        self.matter={'rho_bar':None, 'delta':None, 'vw':None}

        # For iteration scheme
        self.metric_a={ 'a':None, }
        self.metric_dt = {'a':None, }
        self.metric_f = {'a':None, 'Hubble':None, 'Hubble_dt':None}

        self.sols = {'Phi':None, 'Pi':None}

        self.Hubble_0 = None
        self.Hubble = None
        self.a = None
        self.Hubble_dt = None



    def to_tau(self, ntau, d = 0):
        """
        From mesh idx to radius
        """
        return self.tau_f + ntau / self.Ntau * (self.tau_i - self.tau_f)
    # ---------Functions of time derivatives-----------------------

    def da_dt(self, ntau):
        return self.metric_a['a'] * self.Hubble_a

    def dPhi_dt(self, ntau, d):
        return (-2 / self.tau_hier[d][ntau] + 2 * self.Hubble_hier[d][ntau]) \
            * (self.Pi_hier[d][ntau] + (self.Phi_hier[d][ntau + 1] - self.Phi_hier[d][ntau - 1]) / (2 * self.dtau_hier[d])) \
            - (2 * self.Hubble_dt_hier[d][ntau] -2 * self.Hubble_hier[d][ntau]**2
               - 3 * 1.5 * self.Hubble_0**2 *self.Omega_m / self.a_hier[d][ntau]) \
            * self.Phi_hier[d][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.delta_hier[d][ntau] \
            - self.lm * self.Phi_hier[d][ntau] / self.tau_hier[d][ntau]**2  \
    + 3 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.vw_hier[d][ntau]

    def est_f(self, ntau, d):
        """
        Estimate rhs for relaxation (removing linear terms on Phi, which is on the left)
        """
        return (-2 / self.tau_hier[d][ntau] + 2*self.Hubble_hier[d][ntau]) * self.Pi_hier[d][ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.delta_hier[d][ntau] \
    + 3 * self.Hubble_0**2 * self.Omega_m / self.a_hier[d][ntau] * self.vw_hier[d][ntau]


    def dPi_dt(self, step, d):
        cdef npy.ndarray[comp_t, ndim=1, mode='c'] dt = npy.zeros(self.Nalms, dtype = self.cdtype)
        if(step == self.h_size[d] - 1):
            dPi_dt_upper_bd (self.Nalms, \
                       self.Phi_hier[d][step], self.Phi_hier[d][step-1],\
                       self.Phi_hier[d][step-2], self.Pi_hier[d][step], \
                       self.vw_hier[d][step], \
                       dt, self.Pi_i, \
                       self.a_hier[d][step], \
                       self.Hubble_hier[d][step], \
                       self.Hubble_0, self.Hubble_dt_hier[d][step], self.Omega_m, \
                       self.dtau_hier[d]
                       )
        elif(step == 0):

            dPi_dt_lower_bd (self.Nalms, \
                             self.Phi_hier[d][step], self.Phi_hier[d][step+1],\
                             self.Phi_hier[d][step+2], self.Pi_hier[d][step], \
                             self.vw_hier[d][step], \
                             dt,
                             self.a_hier[d][step], \
                             self.Hubble_hier[d][step], \
                             self.Hubble_0, self.Hubble_dt_hier[d][step], self.Omega_m, \
                             self.dtau_hier[d]
                             )
        else:
            dPi_dt (self.Nalms, \
                       self.Phi_hier[d][step], self.Phi_hier[d][step+1],\
                       self.Phi_hier[d][step-1], self.Pi_hier[d][step], \
                       self.vw_hier[d][step], \
                       dt,
                       self.a_hier[d][step], \
                       self.Hubble_hier[d][step], \
                       self.Hubble_0, self.Hubble_dt_hier[d][step], self.Omega_m, \
                       self.dtau_hier[d]
                       )

        return dt

    # --------------------updating other fields-----------
    def update_other_field(self, ntau):
        self.Hubble_a = self.Hubble_0 * self.metric_a['a']\
            * npy.sqrt(self.Omega_m * self.metric_a['a']**(-3) + self.Omega_L )
        self.Hubble_dt_a = self.Hubble_0**2 * self.metric_a['a']**2 * self.Omega_L \
            - self.Hubble_0**2 * self.Omega_m / (2*self.metric_a['a'])

        self.metric_f['Hubble'][ntau] = self.Hubble_a
        self.metric_f['Hubble_dt'][ntau] = self.Hubble_dt_a

    #-----------Time advancing----------------------------------
    # RK2
    def rk2_time_advance_step(self, ntau, dtau):
        if(ntau == 0):
            return

        #----------------first phase-----------------
        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)

        for field in self.metric_a:
            self.metric_a[field] += self.metric_dt[field] * dtau
            self.metric_f[field][ntau-1] = self.metric_f[field][ntau] + 0.5 * dtau * self.metric_dt[field]

        self.update_other_field(ntau-1)


        #------------------second phase--------------
        ntau = ntau - 1

        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)


        for field in self.metric_a:
            self.metric_f[field][ntau] += 0.5 * dtau * self.metric_dt[field]
            self.metric_a[field] = self.metric_f[field][ntau].copy()

        self.update_other_field(ntau)


    def est_errors(self, d, indent = ''):
        """
        Print error information. It is a little expensive!
        """
        err = 0
        rel_err = 0
        mag = 0
        max_step = -1

        for step in reversed(range(1, self.h_size[d]-1)):
            #self.est_angle_lap(step, d)

            rhs = self.dPhi_dt(step, d) + self.rhs_hier[d][step]
            lhs = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                   - 2.0 * self.Phi_hier[d][step ])/ self.dtau_hier[d]**2

            bak = err
            real_err = npy.abs(hp.alm2map((rhs - lhs).astype(npy.cdouble), self.nside))
            err = npy.max([real_err.max(), err])
            rel_err += npy.linalg.norm(real_err)**2
            mag += npy.linalg.norm(hp.alm2map(lhs.astype(npy.cdouble), self.nside))**2
            if(err > bak): max_step = step

        rel_err = npy.sqrt(rel_err / mag)
        for field in self.sols:
            if field == 'Phi':
                print(indent+'Max error for field '+field+' is '+str(err) + ' at step '+str(max_step))
                print(indent+'Relative error of the L2 norm is ' + str(rel_err), flush=True)

    def relax(self, d, nsteps):
        while(nsteps > 0):
            nsteps -= 1
            for field in self.sols:
                #for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                for step in  range(self.h_size[d] - 2, 0, -1):
                    if field == 'Phi':
                        # Newton relax Phi
                        relax_Phi (self.Nalms, \
                                   self.Phi_hier[d][step+1], \
                                   self.Phi_hier[d][step-1], self.Pi_hier[d][step], \
                                   self.delta_hier[d][step], self.vw_hier[d][step], \
                                   self.rhs_hier[d][step], self.Phi_hier[d][step],
                                   self.lm, self.tau_hier[d][step], self.a_hier[d][step], \
                                   self.Hubble_hier[d][step], \
                                   self.Hubble_0, self.Hubble_dt_hier[d][step], self.Omega_m, \
                                   self.dtau_hier[d])

                    elif field == 'Pi':
                        # Time integrate Pi
                        dt = self.dPi_dt(step + 1, d)
                        self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                            + dt * self.dtau_hier[d]
                        dt += self.dPi_dt(step, d)
                        self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                            + 0.5 * self.dtau_hier[d] * dt



    def generate_rl(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.rl_hier[d][step] = (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                 - 2.0 * self.Phi_hier[d][step ])/ (self.dtau_hier[d])**2                    \
                 - self.dPhi_dt(step, d)



    def update_Pi(self, d):
        """
        Since Pi has directly dependence on Phi dot, this will
        generate a new Pi whenever the value of Phi
        """
        for step in  range(self.h_size[d] - 2, -1, -1):
            dt = self.dPi_dt(step + 1, d)
            self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                + dt * self.dtau_hier[d]
            dt += self.dPi_dt(step, d)
            self.Pi_hier[d][step] = self.Pi_hier[d][step+1] \
                + 0.5 * self.dtau_hier[d] * dt

    def update_rhs(self, d):
        for step in  range(self.h_size[d] - 2, 0, -1):
            self.rhs_hier[d][step] += (self.Phi_hier[d][step + 1] + self.Phi_hier[d][step - 1] \
                                 - 2.0 * self.Phi_hier[d][step ])/ (self.dtau_hier[d])**2\
                                 - self.dPhi_dt(step, d) - self.rl_hier[d][step]

    def clear_field(self, data):
        for d in range(self.depth):
            data[d].fill(0)

    def MG(self):
        """
        V-cycle multigrid. Starting from relaxing the finest grid. Might be able to
        get speed-up when using more complicaed cycles with proper stop critiria
        """
        n_vcycles = self.n_vcycles
        while(n_vcycles >0):
            n_vcycles -= 1
            print('Starting v-cycle '+ str(self.n_vcycles - n_vcycles), flush=True)

            if(self.depth == 1):
                upper_stroke = range(1)
            else:
                upper_stroke = range(self.depth - 1)
            for d in upper_stroke:
                self.relax(d, int(self.npre ))

                if(d == 0 or self.verbose == True):
                    print('    At level '+str(d)+', after '+str( int(self.npre) )+' relaxations', flush=True)
                    self.est_errors(d, '    ')

                if(d < self.depth - 1):
                    self.generate_rl(d)

                    #self.hier_restrict(self.Pi_hier,d)
                    self.Pi_bd_restrict(d)

                    self.hier_restrict(self.rl_hier,d)
                    self.hier_restrict(self.Phi_hier,d)

                    self.update_Pi(d+1)
                    self.hier_restrict(self.rhs_hier, d)
                    self.update_rhs(d+1)
            self.clear_field(self.rl_hier)
            for d in range(self.depth - 1, 0, -1):
                self.relax(d, int(self.npost ) )

                if(self.verbose == True):
                    print('    At level '+str(d)+', after '+str( int(self.npost ) )+' relaxations', flush=True)
                    self.est_errors(d, '    ')
                if(d > 0):
                    self.hier_restrict(self.Phi_hier, d-1, self.rl_hier)
                    self.rl_hier[d] = - self.rl_hier[d] + self.Phi_hier[d]
                    self.hier_prolong(self.rl_hier, d)
                    self.Phi_hier[d-1] += self.rl_hier[d-1]
                    self.update_Pi(d-1)
            self.clear_field(self.rhs_hier)
            self.clear_field(self.rl_hier)

        self.relax(0, int(self.npre ))

    def init(self):
        """
        Time evolving scale factor to get a, H, and dH_dt data at every light-cone grid
        """
        #set-up time evolution of a and Hubble and Hubble_dt
        self.update_other_field(self.Ntau)
        for step in reversed(range(1, self.Ntau + 1)):
            self.rk2_time_advance_step(step, (self.tau_i - self.tau_f) / self.Ntau)

        #set-up the initial guess
        for step in  reversed(range(1, self.Ntau)):
            for field in self.sols:
                if(field == 'Phi'):
                    self.sols[field][step] = self.sols[field][self.Ntau] \
                        + (self.Ntau - step) / (self.Ntau) \
                        * (self.sols[field][0] - self.sols[field][self.Ntau])

        # for step in  reversed(range(1, self.Ntau)):
        #     for field in self.sols:
        #         if(field == 'Phi'):
        #             self.sols[field][step] = npy.ascontiguousarray(\
        #                 1.5 * self.Hubble_0**2 * self.Omega_m / self.metric_f['a'][step] \
        #                 * hp.sphtfunc.alm2map((hp.sphtfunc.map2alm(
        #                     self.matter['delta'][step], lmax = self.lmax, iter = self.alm_iter)
        #                                        * npy.nan_to_num(1/ self.lm, posinf=0) ).astype(npy.cdouble)
        #                                       , nside = self.nside) \
        #                                                            * self.to_tau(step, 0)**2, dtype=self.pdtype)
    # Only restricing boundary data for Pi
    # used by integration
    def Pi_bd_restrict(self, d):
        if( d == self.depth - 1):
            return
        self.Pi_hier[d+1][0] = self.Pi_hier[d][0]
        if(self.h_size[d] % 2 == 0):
            self.Pi_hier[d+1][-1] = self.Pi_hier[d][-2]
        else:
            self.Pi_hier[d+1][-1] = self.Pi_hier[d][-1]


    # Restricting data from depth d to d + 1 (fine to coarse)
    def hier_restrict(self, data, d, tdata = None):
        if( d == self.depth - 1):
            return
        if(tdata == None): tdata = data
        tdata[d+1][1:-1] = 0.5 * data[d][2:-2:2] + 0.25 * (data[d][1:-3:2] + data[d][3:-1:2])
        tdata[d+1][0] = data[d][0]
        if(self.h_size[d] % 2 == 0):
            tdata[d+1][-1] = data[d][-2]
        else:
            tdata[d+1][-1] = data[d][-1]


    # Prolonging data from depth d to d-1 (coarse to fine)
    def hier_prolong(self, data, d, tdata = None):
        if( d == 0):
            return
        if(tdata == None): tdata = data
        tdata[d-1][2:-2:2] = data[d][1:-1]
        tdata[d-1][1:-1:2] = 0.5 * (data[d][0:-1] + data[d][1:])
        tdata[d-1][0] = data[d][0]

    def to_alm(self, data, array_data = True, pixwin=None):

        if(array_data == True):
            if(pixwin == None):
                return npy.ascontiguousarray( [hp.sphtfunc.map2alm(
                    data[n], lmax = self.lmax, iter = self.alm_iter) \
                                               for n in range(len(data))],
                                              dtype=self.cdtype)
            elif(pixwin == 'CIC'):
                return npy.ascontiguousarray( [hp.almxfl(hp.sphtfunc.map2alm(
                    data[n], lmax = self.lmax, iter = self.alm_iter), 1 / hp.pixwin(self.nside)**2 ) \
                                               for n in range(len(data))],
                                              dtype=self.cdtype)
            elif(pixwin == 'NGP'):
                return npy.ascontiguousarray( [hp.almxfl(hp.sphtfunc.map2alm(
                    data[n], lmax = self.lmax, iter = self.alm_iter), 1 / hp.pixwin(self.nside) ) \
                                               for n in range(len(data))],
                                              dtype=self.cdtype)
            else:
                raise ValueError('pixwin type' + str(pixwin))
        else:
            if(pixwin == None):
                return npy.ascontiguousarray(hp.sphtfunc.map2alm(
                    data, lmax = self.lmax, iter = self.alm_iter),
                                             dtype=self.cdtype)
            elif(pixwin == 'CIC'):
                return npy.ascontiguousarray(hp.almxfl(hp.sphtfunc.map2alm(
                    data, lmax = self.lmax, iter = self.alm_iter), 1 / hp.pixwin(self.nside)**2 ),
                                             dtype=self.cdtype)
            elif(pixwin == 'NGP'):
                return npy.ascontiguousarray(hp.almxfl(hp.sphtfunc.map2alm(
                    data, lmax = self.lmax, iter = self.alm_iter), 1 / hp.pixwin(self.nside) ),
                                             dtype=self.cdtype)
            else:
                raise ValueError('pixwin type' + str(pixwin))


    def to_real(self, data, array_data = True):
        if(array_data == True):
            return npy.ascontiguousarray([hp.sphtfunc.alm2map(
                data[n].astype(npy.cdouble), nside = self.nside)
                                          for n in range(len(data))],
                                         dtype=self.pdtype)
        else:
            return npy.ascontiguousarray(hp.sphtfunc.alm2map(
                data.astype(npy.cdouble), nside = self.nside), dtype=self.pdtype)


    def build_hier(self, alm_form = False):

        # Initializing finest hiers
        self.Phi_hier = [None] * self.depth
        self.Pi_hier = [None] * self.depth
        self.delta_hier = [None] * self.depth
        self.vw_hier = [None] * self.depth
        self.a_hier = [None] * self.depth
        self.Hubble_hier = [None] * self.depth
        self.Hubble_dt_hier = [None] * self.depth
        self.lm_hier = [None] * self.depth
        self.tau_hier = [None] * self.depth
        self.dtau_hier = [None] * self.depth
        self.h_size = [None] * self.depth

        self.rhs_hier = [None] * self.depth
        self.rl_hier = [None] * self.depth

        # Calculating tau (r) hier and dtau hier
        self.h_size[0] = self.Ntau  + 1
        n = self.h_size[0] #h_size includes both ends
        for d in range(self.depth-1):
            if(n %2 == 0):
                n = int(n /2)
            else:
                n = int((n + 1) / 2)
            self.h_size[d+1] = n

        for d in range(self.depth):
            self.tau_hier[d] = npy.array([self.to_tau(i, d) for i in range(self.h_size[d])])
            self.dtau_hier[d] = (self.tau_i - self.tau_f) / self.Ntau * 2**d


        if(alm_form == True):
            self.Phi_hier[0] = self.sols['Phi']
            self.Pi_hier[0] = self.sols['Pi']
        else:
            self.Phi_hier[0] = self.to_alm(self.sols['Phi'])
            self.Pi_hier[0] = self.to_alm(self.sols['Pi'])

        self.delta_hier[0] = self.to_alm(self.matter['delta'], pixwin=self.depo_method)
        self.vw_hier[0] = self.to_alm(self.matter['vw'], pixwin=self.depo_method)

        if self.sm_dx > 0:
            print('Smothing delta and v with sm_dx ' + str(self.sm_dx), flush=True)

            for n in range(self.Ntau + 1):
                self.delta_hier[0][n] = hp.smoothalm(self.delta_hier[0][n],
                                                     fwhm = self.sm_dx/self.tau_hier[0][n]/2,
                                                     inplace=False,
                                                     verbose=False)
                self.vw_hier[0][n] = hp.smoothalm(self.vw_hier[0][n],
                                                  fwhm = self.sm_dx/self.tau_hier[0][n]/2,
                                                  inplace=False,
                                                  verbose=False)

        self.a_hier[0] = self.metric_f['a']
        self.Hubble_hier[0] = self.metric_f['Hubble']
        self.Hubble_dt_hier[0] = self.metric_f['Hubble_dt']

        self.rhs_hier[0] = npy.zeros(self.Phi_hier[0].shape, dtype = self.cdtype)
        self.rl_hier[0] = npy.zeros(self.Phi_hier[0].shape, dtype = self.cdtype)


        # Allocating space for hiers
        # and initializing data through hier_restrict
        self.h_size[0] = self.Ntau  + 1
        n = self.h_size[0] #h_size includes both ends
        self.h_size[0] = n
        for d in range(self.depth-1):
            if(n %2 == 0):
                n = int(n /2)
            else:
                n = int((n + 1) / 2)

            self.h_size[d+1] = n

            self.Phi_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)
            self.hier_restrict(self.Phi_hier, d)

            self.Pi_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)

            self.delta_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)
            self.hier_restrict(self.delta_hier, d)

            self.vw_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)
            self.hier_restrict(self.vw_hier, d)

            self.rhs_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)
            self.rl_hier[d+1] = npy.zeros((n, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)


            self.a_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.a_hier, d)

            self.Hubble_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.Hubble_hier, d)

            self.Hubble_dt_hier[d+1] = npy.zeros(n)
            self.hier_restrict(self.Hubble_dt_hier, d)




    #-------------Initializations------------------------------------

    def Phi_Pi_gen(self, delta, vw, dvw_dt, H, a, r):
        Pi_zel = -1.5 * H * vw

        PPi_zel = (Pi_zel + vw * H + dvw_dt) * 1.5*H


        H0 = self.Hubble_0
        rhs = r**2 * \
            (1.5 * H0**2 * self.Omega_m / a * delta \
             - (2 * Pi_zel + r * PPi_zel) / r )
        self.rhs = self.to_alm(rhs, array_data = False)

        self.coeffs = 2 * r**2 * (1.5 * self.Hubble_0**2 * self.Omega_m / a )

        self.Phi_zel = self.rhs / (self.coeffs + self.lm)

        self.Pi_zel = self.to_alm(Pi_zel, array_data = False)

        return self.Phi_zel, self.Pi_zel

    def init_from_matter(self, z_i_in, r_max_in, delta, vw, Params, \
                         r_min_in):
        """
        Initialize from delta and vw fields only from Zel' approximation
        Still under construction
        """
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.tau_f = r_min_in

        self.Ntau = delta.shape[0] - 2

        #self.ntau_f = int(npy.round(self.tau_f / (self.tau_i / self.Ntau)))

        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        #descritation index ranging from 0 to Ntau
        for field in self.sols:
            self.sols[field] = npy.zeros((self.Ntau + 1, hp.Alm.getsize(self.lmax)), dtype = self.cdtype)


        self.matter['delta'] = delta[0:self.Ntau+1].copy()
        self.matter['vw'] = vw[0:self.Ntau+1].copy()
        # Need to substract extra 2 padding grid

        #self.sols['Phi'][-1] = Phi_i_in.copy()
        #self.sols['Pi'][-1] = Pi_i_in.copy()


        #self.sols['Phi'][0] = Phi_f_in.copy()
        #self.sols['Pi'][0] = Pi_f_in.copy()
        #self.sols['Pi'][0] = npy.zeros(Pi_i_in.shape) # Do not need Pi_f

        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][-1] = 1 / (1 + z_i_in)

        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!", flush=True)

        self.init()

        for step in  range(self.Ntau , -1, -1):
            self.sols['Phi'][step], self.sols['Pi'][step] = \
                self.Phi_Pi_gen(delta[step ], vw[step ], \
                                -(vw[step  + 1] - vw[step - 1]) \
                                /  (2 * (self.tau_i - self.tau_f)/self.Ntau), \
                                self.metric_f['Hubble'][step], \
                                self.metric_f['a'][step], self.to_tau(step))
            #self.sols['Phi'][step] -= self.sols['Phi'][step].mean()


        self.build_hier(alm_form = True)

        self.update_Pi(0)

        if(self.grid == 'DH'):
            self.grid = 'DH2'


    def init_from_slice(self, z_i_in, r_max_in, delta_in, vw_in, Phi_i_in, Pi_i_in, \
                        Params, r_min_in, Phi_f_in, Omega_i=None, depo_method='NGP',
                        sm_dx=0):
        """Sample function of setting-up initial data
        Parameters
        ----------
        z_i_in : float
            initial z
        r_max_in: float
            initial radial distance at z
        delta_in: float, shape(N+2, NPIX)
        delta_in: float, shape(N+2, NPIX)
        Phi_i_in: float, shape(NPIX)
        Pi_i_in: float, shape(NPIX)
        Paras: list
            Cosmological parameters, needs to include 'h', 'Omega_m' and 'Omega_L' term
        r_min_in: float
            Final radial distance at z_f
        Phi_f_in: float, shape(NPIX)
        Returns
        -------
        void
        """
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.tau_f = r_min_in

        self.Ntau = delta_in.shape[0] - 2

        self.pdtype = Phi_i_in.dtype
        self.depo_method = depo_method
        self.sm_dx = sm_dx

        if self.pdtype == npy.single:
            self.cdtype = npy.csingle
        elif self.pdtype == npy.double:
            self.cdtype = npy.cdouble
        else:
            raise ValueError('Data type' + str(self.pdtype)
                             + ' of Phi is not supported!')

        print('Set data type as ' + str(self.pdtype), flush=True)

        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        #descritation index ranging from 0 to Ntau
        for field in self.sols:
            self.sols[field] = npy.zeros((self.Ntau + 1, self.Npix), dtype=self.pdtype)

        self.matter['delta'] = delta_in[0:self.Ntau+1]
        self.matter['vw'] = vw_in[0:self.Ntau+1]
        # Need to substract extra 2 padding grid

        self.sols['Phi'][-1] = Phi_i_in.copy()
        self.sols['Pi'][-1] = Pi_i_in.copy()


        self.sols['Phi'][0] = Phi_f_in.copy()
        #self.sols['Pi'][0] = Pi_f_in.copy()
        self.sols['Pi'][0] = npy.zeros(Pi_i_in.shape, dtype=self.pdtype) # Do not need Pi_f


        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][-1] = 1 / (1 + z_i_in)

        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!", flush=True)

        self.init()

        # Omega_i = npy.ascontiguousarray(npy.load(
        #     "/media/chris/3b7ae93c-9459-4858-9b27-3209d1805b9a/draft_data/Metric_recon/con_test_256_phi/Omega_nl.npy")
        #                                 , dtype=self.pdtype)
        if(Omega_i is None):
            Omega_i = self.metric_f['Hubble'][self.Ntau] * \
                Phi_i_in * (((Params['Omega_m'] * (1 + z_i_in)**3 ) \
                             / (Params['Omega_m'] * (1 + z_i_in)**3 + Params['Omega_L']))**0.55 - 1)

        self.Pi_i = self.to_alm(self.sols['Pi'][-1], array_data=False)
        self.sols['Pi'][-1] =  Omega_i


        self.build_hier()

        self.update_Pi(0)


    def build_lcmetric(self):
        self.MG()
        # Transfering resules back to real fields
        # Notice here the initial and final slices are also transfered
        # to maintain smoothness when caclulating time derivatives.
        self.sols['Phi'] = self.to_real(self.Phi_hier[0])
        self.sols['Pi'] = self.to_real(self.Pi_hier[0])

        del self.rl_hier
        del self.rhs_hier
        #del self.Phi_hier
        #del self.Pi_hier

        del self.delta_hier
        del self.vw_hier
