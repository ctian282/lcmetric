import numpy as npy
import healpy as hp
import numexpr as ne
import numba as nb

import lcmetric.utils as ut

#Relaxation scheme with pure alms by healpix
class Lightcone:

    # does not really use rho_bar, replaced by H
    matter={'rho_bar':None, 'delta':None, 'vw':None}


    # For iteration scheme
    metric_a={ 'a':None, }
    metric_dt = {'a':None, }
    metric_f = {'a':None, 'Hubble':None, 'Hubble_dt':None}

    sols = {'Phi':None, 'Pi':None}

    

    Hubble_0 = None
    Hubble = None
    a = None
    Hubble_dt = None

    angle_lap = None

    nside = None

    theta_list = None

    phi_list = None
    
    def  __init__(self, delta_in, vw_in, nside, lmax_in = None, \
                  epsilon_in = 1e-6, tol_in= 1e-7, niter_in = 100, use_SOR = False, grid = 'healpy',
                  alm_iter = 100):

        self.matter['delta'] = delta_in.copy()
        self.matter['vw'] = vw_in.copy()
        # Need to substract extra 2 padding grid 
        self.Ntau = delta_in.shape[0] - 2
        self.nside = nside
        self.Npix = 12 * nside**2
        self.lmax = lmax_in
        self.alm_iter = alm_iter
        self.use_SOR = use_SOR
        self.grid = grid
        
        
        self.epsilon = epsilon_in
        self.tol = tol_in
        self.niter = niter_in
        
        # descritation index ranging from 0 to Ntau
        for field in self.metric_f:
            self.metric_f[field] = npy.zeros(self.Ntau + 1)

        for field in self.sols:
            self.sols[field] = npy.zeros((self.Ntau + 1, 12*self.nside**2))

            
    def to_tau(self, ntau):
        return (ntau) / self.Ntau * self.tau_i

    # ---------Functions of time derivatives-----------------------
        
    def da_dt(self, ntau):
        #return 0
        return self.metric_a['a'] * self.Hubble_a

    def dPhi_dt(self, ntau):
        return (-2 / self.tau_list[ntau] + 2*self.Hubble[ntau]) * self.Pi[ntau] \
            - (2 * self.Hubble_dt[ntau] -2 * self.Hubble[ntau]**2) \
            * self.Phi[ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m / self.a[ntau] * self.delta[ntau] \
            - self.Phi[ntau] * self.lm / self.tau_list[ntau]**2 \
    + 3 * self.Hubble_0**2 * self.Omega_m * self.vw[ntau]

    
    def dPi_dt(self, ntau):
        if (ntau == self.Ntau):
            Omega_dot = (self.Phi[ntau] + self.Phi[ntau-2] - 2*self.Phi[ntau-1]) \
                / (self.tau_i / self.Ntau)**2
            Omega = -(self.Phi[ntau] - self.Phi[ntau-1]) / (self.tau_i / self.Ntau)
        else:
            Omega_dot = (self.Phi[ntau+1] + self.Phi[ntau-1] - 2*self.Phi[ntau]) \
                / (self.tau_i / self.Ntau)**2
            Omega = -(self.Phi[ntau + 1] - self.Phi[ntau-1]) / (2.0 * self.tau_i / self.Ntau)


            
        return - Omega_dot - 3 * self.Hubble[ntau] * Omega \
            - 2 * self.Hubble[ntau] * self.Pi[ntau] \
            - (2 * self.Hubble_dt[ntau] + self.Hubble[ntau]**2) * self.Phi[ntau] \
            + 1.5 * self.Hubble_0**2 * self.Omega_m * self.vw[ntau]
                       
    
    # --------------------updating other fields-----------
    def update_other_field(self, ntau):
        self.Hubble_a = self.Hubble_0 * self.metric_a['a']\
            * npy.sqrt(self.Omega_m * self.metric_a['a']**(-3) + self.Omega_L )
        self.Hubble_dt_a = self.Hubble_0**2 * self.metric_a['a']**2 * self.Omega_L \
            - self.Hubble_0**2 * self.Omega_m / (2*self.metric_a['a'])

        # self.Hubble = 0
        # self.Hubble_dt = 0
        
        self.metric_f['Hubble'][ntau] = self.Hubble_a
        self.metric_f['Hubble_dt'][ntau] = self.Hubble_dt_a

        
    #-----------Time advancing----------------------------------
    # RK2
    def rk2_time_advance_step(self, ntau, dtau):
        #----------------first phase-----------------
        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)

        for field in self.metric_a:
            self.metric_a[field] += self.metric_dt[field] * dtau
            self.metric_f[field][ntau-1] = self.metric_f[field][ntau] + 0.5 * dtau * self.metric_dt[field] 

        self.update_other_field(ntau-1)

        if(ntau == 1):
            for field in self.metric_a:
                self.metric_f[field][0] = self.metric_a[field].copy()
            return
        
        #------------------second phase--------------
        ntau = ntau - 1

        for field in self.metric_a:
            self.metric_dt[field] = eval('self.d'+field+'_dt')(ntau)


        for field in self.metric_a:
            self.metric_f[field][ntau] += 0.5 * dtau * self.metric_dt[field] 
            self.metric_a[field] = self.metric_f[field][ntau].copy()

        self.update_other_field(ntau)


    def est_errors(self):
        errs = {'Phi':0.0, 'Pi':0.0, 'Omega':0.0}
        rel_err = 0
        mag = 0
        for field in self.sols:
            for step in reversed(range(self.ntau_f + 1, self.Ntau)):
                #self.est_angle_lap(step)
                if field == 'Phi':
                    rhs = eval('self.d'+field+'_dt')(step)
                    lhs = (self.Phi[step + 1] + self.Phi[step - 1] \
                           - 2.0 * self.Phi[step ])/ (self.tau_i / self.Ntau)**2
                    if(self.grid=='DH2'):
                        real_err = hp.alm2map(rhs - lhs, iter = self.alm_iter)
                        errs[field] = npy.max([real_err.max(), errs[field]])
                        rel_err += npy.linalg.norm(real_err)**2
                        mag += npy.linalg.norm(hp.alm2map(lhs, iter = self.alm_iter))**2
                    else:
                        errs[field] = npy.max([npy.abs( lhs - rhs).max(), errs[field]])
                        rel_err += npy.linalg.norm( (lhs - rhs) )**2
                        mag += npy.linalg.norm(lhs)**2                    
        rel_err = npy.sqrt(rel_err / mag)
        for field in self.sols:
            if field == 'Phi':
                print('Max error for field '+field+' is '+str(errs[field]))
                print('Relative error of the L2 norm is ' + str(rel_err))

                        
    def est_angle_lap(self, ntau):
        if(self.angle_lap_type == 'FD'):
            # self.angle_lap = ut.fderv1(self.Phi[ntau], self.dtheta, 0)
            # self.angle_lap[1,:] = (self.Phi[ntau, 2, :] - self.Phi[ntau, 1, :]) / self.dtheta
            # self.angle_lap[self.nlat-1,:] = (self.Phi[ntau, self.nlat-1, :] - self.Phi[ntau, self.nlat-2, :]) \
            #     / self.dtheta
            
            # self.angle_lap *= npy.sin(self.theta_list)[:,None]
            
            # self.angle_lap = ut.fderv1(self.angle_lap, self.dtheta, 0)
            # self.angle_lap[1,:] = (self.angle_lap[2, :] - self.angle_lap[1, :]) / self.dtheta
            # self.angle_lap[self.nlat-1,:] = (self.angle_lap[self.nlat-1, :] - self.angle_lap[self.nlat-2,:]) \
            #     / self.dtheta
            
            # self.angle_lap /= npy.sin(self.theta_list)[:, None]

            self.angle_lap =  ut.fderv1(self.Phi[ntau], self.dtheta, 0)
            self.angle_lap[1,:] = (self.Phi[ntau, 2, :] - self.Phi[ntau, 1, :]) / self.dtheta
            self.angle_lap[self.nlat-1,:] = (self.Phi[ntau, self.nlat-1, :] - self.Phi[ntau, self.nlat-2, :]) \
                 / self.dtheta
            self.angle_lap /= npy.tan(self.theta_list)[:,None]

            ddtheta =  ut.fderv2(self.Phi[ntau], self.dtheta, 0)
            ddtheta[1,:] = (self.Phi[ntau, 3, :] + self.Phi[ntau, 1, :] \
                            - 2*self.Phi[ntau, 2, :]) / self.dtheta**2
            ddtheta[self.nlat - 1,:] = (self.Phi[ntau, self.nlat - 3, :] + self.Phi[ntau, self.nlat - 1, :] \
                            - 2*self.Phi[ntau, self.nlat - 2, :]) / self.dtheta**2

            self.angle_lap += ddtheta

            
            self.angle_lap += ut.fderv2(self.Phi[ntau], self.dphi, 1) / npy.sin(self.theta_list)[:,None]**2
            self.angle_lap[0,:] = 0
            self.angle_lap /= self.tau_list[ntau]**2
        else:
            alm=self.sh_grid[ntau].expand(lmax_calc=self.lmax)
            alm.coeffs*=self.lm
            self.angle_lap = alm.expand(grid=self.grid, extend = self.extend).data / self.tau_list[ntau]**2

        #return self.angle_lap
        #Self.angle_lap = 0
            
    def iteration(self):
        nit = self.niter
        rho_Jac = 1 - (npy.pi / 2 / (self.Ntau - self.ntau_f))**2
        w = 2 / (1+npy.sqrt(1-rho_Jac**2))
        while(nit > 0):
            if(nit % 200 == 0):
                print('For iteration '+str(nit))
                self.est_errors()
            nit -= 1
            #raise ValueError('A very specific bad thing happened.')
            if(self.use_SOR == False):
                for field in self.sols:
                #for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                    for step in  range(self.Ntau-1, self.ntau_f, -1):
                        if field == 'Phi':
                            #self.est_angle_lap(step)
                            self.Phi[step] = (self.Phi[step + 1] + self.Phi[step - 1] - \
                                (self.tau_i / self.Ntau)**2 * eval('self.d'+field+'_dt')(step)) / 2
                        elif field == 'Pi':
                            dt = eval('self.d'+field+'_dt')(step + 1)
                            self.Pi[step] = self.Pi[step+1] \
                                + dt * (self.tau_i / self.Ntau)
                            dt += eval('self.d'+field+'_dt')(step)
                            self.Pi[step] = self.Pi[step+1] \
                                + 0.5 * self.tau_i / self.Ntau * dt
            else:
                for field in self.sols:
                    #for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
                    for step in  range(self.Ntau-1, self.ntau_f, -1):
                        #self.est_angle_lap(step)
                        if field == 'Phi':
                            self.Phi[step] +=  \
                            w * 0.5 * (self.Phi[step + 1] + self.Phi[step - 1] - 2 * self.Phi[step] - \
                                (self.tau_i / self.Ntau)**2 * eval('self.d'+field+'_dt')(step)) 

                        elif field == 'Pi':
                            dt = eval('self.d'+field+'_dt')(step + 1)
                            self.Pi[step] = self.Pi[step+1] \
                                + dt * (self.tau_i / self.Ntau)
                            dt += eval('self.d'+field+'_dt')(step)
                            self.Pi[step] = self.Pi[step+1] \
                                + 0.5 * self.tau_i / self.Ntau * dt
                w = ((-4 + w * rho_Jac**2) / (-4 + (1+ w) * rho_Jac**2)  - w) * 0.01 + w 
                  
    def update_Pi(self):
        for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
            dt = eval('self.dPi_dt')(step + 1)
            self.Pi[step] = self.Pi[step+1] \
                + dt * (self.tau_i / self.Ntau)
            dt += eval('self.dPi_dt')(step)
            self.Pi[step] = self.Pi[step+1] \
                + 0.5 * self.tau_i / self.Ntau * dt

        
    def init(self):
        #set-up time evolution of a and Hubble and Hubble_dt
        self.update_other_field(self.Ntau)
        for step in reversed(range(1, self.Ntau + 1)):
            self.rk2_time_advance_step(step, self.tau_i / self.Ntau)

        #set-up the initial guess
        for step in  reversed(range(self.ntau_f + 1, self.Ntau)):
            for field in self.sols:
                if(field == 'Phi'):
                    self.sols[field][step] = self.sols[field][self.Ntau] \
                        + (self.Ntau - step) / (self.Ntau - self.ntau_f) \
                        * (self.sols[field][self.ntau_f] - self.sols[field][self.Ntau])


            
    #-------------Initializations------------------------------------
    def init_from_slice(self, z_i_in, r_max_in, Phi_i_in, Pi_i_in, Params, \
                        z_f_in, r_min_in, Phi_f_in, Pi_f_in):

        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.a_f = 1 / (1 + z_f_in)

        self.tau_f = r_min_in

        self.ntau_f = int(npy.round(self.tau_f / (self.tau_i / self.Ntau)))

        self.sols['Phi'][self.Ntau] = Phi_i_in.copy()
        self.sols['Pi'][self.Ntau] = Pi_i_in.copy()


        self.sols['Phi'][self.ntau_f] = Phi_f_in.copy()
        self.sols['Pi'][self.ntau_f] = Pi_f_in.copy()

        self.theta_list, self.phi_list = hp.pix2ang(self.nside, range(12*self.nside**2))

        if(self.lmax == None ): self.lmax = 3*self.nside - 1
        
        self.lm = hp.Alm.getlm(self.lmax, npy.array(range( int(1+self.lmax + 0.5 * self.lmax*(1+self.lmax)) )))
        self.lm = -self.lm[0] *(self.lm[0] + 1)
        
        

        self.tau_list = npy.array([self.to_tau(i) for i in range(self.Ntau+1)])
        
        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][self.Ntau] = 1 / (1 + z_i_in)

        # self.metric_a['a'] = 1
        # self.metric_f['a'][self.Ntau] = 1

        
        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        if( abs(self.Omega_m + self.Omega_L - 1.0) > 0.05 ):
            print("Warning! the total energy fraction is deviating from 1!!!")

        self.scheme = 'iteration'

        self.init()



        self.Phi = npy.array( [hp.sphtfunc.map2alm( \
                            self.sols['Phi'][n],lmax = self.lmax, iter = self.alm_iter) \
                               for n in range(self.Ntau+1)])
        self.Pi = npy.array( [hp.sphtfunc.map2alm( \
                            self.sols['Pi'][n], lmax = self.lmax, iter = self.alm_iter) \
                               for n in range(self.Ntau+1)])
        self.delta = npy.array( [hp.sphtfunc.map2alm( \
                            self.matter['delta'][n], lmax = self.lmax, iter = self.alm_iter) \
                              for n in range(self.Ntau+1)])
        self.vw = npy.array( [hp.sphtfunc.map2alm( \
                            self.matter['vw'][n], lmax = self.lmax, iter = self.alm_iter) \
                                 for n in range(self.Ntau+1)])
        self.a = self.metric_f['a']
        self.Hubble = self.metric_f['Hubble']
        self.Hubble_dt = self.metric_f['Hubble_dt']

        self.update_Pi()
        
    def Phi_Pi_gen(self, delta, vw, dvw_dt, H, a, r):
        Pi_zel = -1.5 * H * vw
        self.Pi_zel = Pi_zel
        PPi_zel = (Pi_zel + vw * H + dvw_dt) * 1.5
        self.PPi_zel = PPi_zel
        H0 = self.Hubble_0
        self.rhs = r**2 * \
            (1.5 * H0**2 * self.Omega_m / a * delta \
             - (2 * Pi_zel + r * PPi_zel) / r )
        self.rhs_pysh = \
            pysh.SHGrid.from_array(self.rhs, grid = self.grid, copy = False)
        self.coeffs = 2 * r**2 * (1.5 * self.Hubble_0**2 * self.Omega_m / a )
        self.alm = self.rhs_pysh.expand(lmax_calc=self.lmax)
        self.alm.coeffs /= self.lm + self.coeffs
        temp_grid = self.grid
        if(temp_grid == 'DH'): temp_grid = 'DH2'
        self.Phi_zel = self.alm.expand(grid=temp_grid, extend = self.extend).data
        return self.Phi_zel, self.Pi_zel
        
    def init_from_delta_vw(self, z_i_in, r_max_in, Params, \
                        z_f_in, r_min_in):
        self.r_max = r_max_in;

        self.tau_i = r_max_in;

        self.a_f = 1 / (1 + z_f_in)

        self.tau_f = r_min_in

        self.ntau_f = int(npy.round(self.tau_f / (self.tau_i / self.Ntau)))

        self.Phi = self.sols['Phi']
        self.Pi = self.sols['Pi']
        self.delta = self.matter['delta']
        self.vw = self.matter['vw']
        self.a = self.metric_f['a']
        self.Hubble = self.metric_f['Hubble']
        self.Hubble_dt = self.metric_f['Hubble_dt']

        self.Hubble_0 = Params['h'] * 100
        self.Omega_m = Params['Omega_m']
        self.Omega_L = Params['Omega_L']

        
        self.tau_list = npy.array([self.to_tau(i) for i in range(self.Ntau+1)])
        
        self.metric_a['a'] = 1 / (1 + z_i_in)
        self.metric_f['a'][self.Ntau] = 1 / (1 + z_i_in)
        

        self.sh_grid = \
            npy.array([pysh.SHGrid.from_array(self.sols['Phi'][n], grid = self.grid, copy = False) \
                       for n in range(self.Ntau+1)])

        if(self.lmax == None ): self.lmax = self.sh_grid[self.Ntau].lmax
        self.lm = npy.array([[ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)],
                             [ [-l*(l+1) for m in range(self.lmax+1)] for l in range(self.lmax+1)]])



        self.init()

        for step in  range(self.Ntau, self.ntau_f-1, -1):
            self.sols['Phi'][step], self.sols['Pi'][step] = \
                self.Phi_Pi_gen(self.delta[step], self.vw[step], \
                                -(self.vw[step + 1] - self.vw[step - 1]) /  (2 * self.tau_i/self.Ntau), \
                                self.Hubble[step], self.a[step], self.tau_list[step])
            self.sols['Phi'][step] -= self.sols['Phi'][step].mean()

        # self.sols['Phi'][self.Ntau], self.sols['Pi'][self.Ntau] = \
        #     self.Phi_Pi_gen(self.delta[self.Ntau], self.vw[self.Ntau], \
        #                -(self.vw[self.Ntau + 1] - self.vw[self.Ntau - 1]) /  (2 * self.tau_i/self.Ntau), \
        #                self.Hubble[self.Ntau], self.a[self.Ntau], self.tau_i)

        # self.sols['Phi'][self.Ntau] -= self.sols['Phi'][self.Ntau].mean()
        
        # self.sols['Phi'][self.ntau_f], self.sols['Pi'][self.ntau_f] = \
        #     self.Phi_Pi_gen(self.delta[self.ntau_f], self.vw[self.ntau_f], \
        #                -(self.vw[self.ntau_f + 1] - self.vw[self.ntau_f - 1]) /  (2 * self.tau_i/self.Ntau), \
        #                self.Hubble[self.ntau_f], self.a[self.ntau_f], self.tau_f)
        # self.sols['Phi'][self.ntau_f] -= self.sols['Phi'][self.ntau_f].mean()
        
        if(self.grid == 'DH'):
            self.grid = 'DH2'
            self.dtheta = npy.radians( npy.abs(self.sh_grid[0].lats()[2] - self.sh_grid[0].lats()[1] ) )
            self.dphi = npy.radians(self.sh_grid[0].lons()[2] - self.sh_grid[0].lons()[1])


    

        self.theta_list = npy.radians(90 - self.sh_grid[0].lats())
        self.phi_list = npy.radians(self.sh_grid[0].lons())
        
        
        
    def build_lcmetric(self):
        self.iteration()
        for n in range(self.ntau_f+1, self.Ntau):
            for field in self.sols:
                self.sols[field][n] = hp.alm2map( \
                                                  eval('self.'+field)[n], self.nside)
