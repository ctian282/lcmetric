import numpy as npy
import scipy as scpy

from nbodykit.source.catalog import Gadget1Catalog
from nbodykit.source.catalog import CSVCatalog

import healpy as hp
import glob
import struct

import lcmetric.metric as met
import lcmetric.clcdensity as dens
import lcmetric.cgeodesic as geo
import lcmetric.clightcone_CIC as lc_CIC
import lcmetric.utils as ut
import lcmetric.cutils as utC


class Lightcone:
    def __init__(self, origin, cosmo_paras, L_snap, N_snap, NR, NSIDE,
                 **keyws):
        # dytpe only govens array data type regarding to any
        # particle data, including their pos, vel, pot, AND
        # any data on lightcone mesh. All other float data
        # is in double
        self.pdtype = keyws.get('dtype', 'f8')

        self.L_snap = L_snap
        self.N_snap = N_snap
        self.NR = NR
        self.NSIDE = NSIDE
        self.NPIX = 12 * self.NSIDE**2

        self.cosmo_paras = cosmo_paras

        self.origin = npy.array(origin, dtype=npy.double)
        self.theta_list, self.phi_list = hp.pix2ang(NSIDE,
                                                    range(12 * NSIDE**2))
        self.theta_list = npy.ascontiguousarray(self.theta_list,
                                                dtype=npy.double)
        self.phi_list = npy.ascontiguousarray(self.phi_list, dtype=npy.double)

        self.dx = npy.array([self.L_snap / self.N_snap] * 3)

        # L unit
        self.L_unit = 3e5
        self.hL_unit = self.L_unit * cosmo_paras['h']

        # setting multigrid parameters
        alm_iter = keyws.get('alm_iter', 50)
        depth = keyws.get('depth', 5)
        n_vcycles = keyws.get('n_vcycles', 100)
        npre = keyws.get('npre', 16)
        npost = keyws.get('npost', 16)
        lmax = keyws.get('lmax', 2 * NSIDE - 1)
        self.met = met.Metric(NSIDE,
                              epsilon=1e-12,
                              grid='healpy',
                              alm_iter=alm_iter,
                              depth=depth,
                              n_vcycles=n_vcycles,
                              npre=npre,
                              npost=npost,
                              lmax=lmax,
                              verbose=False)

    # 1/H(z)
    def Hint(self, z):
        return 1 / (self.cosmo_paras['h'] * 100 *
                    npy.sqrt(self.cosmo_paras['Omega_m'] *
                             (1 + z)**3 + self.cosmo_paras['Omega_L']))

    # H(z)
    def H(self, z):
        return (self.cosmo_paras['h'] * 100 / (1 + z) *
                npy.sqrt(self.cosmo_paras['Omega_m'] *
                         (1 + z)**3 + self.cosmo_paras['Omega_L']))

    # integrate 1/H(z) - r, to inversely solve z for given r
    def inverse_Hint(self, z, r):
        return npy.abs(scpy.integrate.quad(self.Hint, 0, z)[0] - r)

    def lensing_kernel(self, r, rs):
        a = 1 / (1 +
                 scpy.optimize.minimize_scalar(self.inverse_Hint, args=(r).x))
        return (rs - r) * r / rs / a

    def read_snap_density(self, path, snap_type, a):
        if (snap_type == 'CSV'):
            # Start reading initial snapshot and setting initial data
            f = CSVCatalog(path, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi'],
                           dtype=self.pdtype)

            f['Position'] = (f['x'][:, None]) * [1/self.hL_unit, 0, 0] \
                + (f['y'][:, None]) * [0, 1/self.hL_unit, 0] \
                + (f['z'][:, None]) * [0, 0, 1/self.hL_unit]

            f.attrs['BoxSize'] = self.L_snap
        elif (snap_type == 'Gadget1'):
            f = Gadget1Catalog(path,
                               columndefs=[
                                   (
                                       'Position',
                                       ('auto', 3),
                                       'all',
                                   ),
                                   (
                                       'GadgetVelocity',
                                       ('auto', 3),
                                       'all',
                                   ),
                                   (
                                       'ID',
                                       'auto',
                                       'all',
                                   ),
                               ])
            f['Position'] *= 1 / self.hL_unit
            f.attrs['BoxSize'] = self.L_snap
        else:
            raise ValueError('Snap data type is not supported!')

        self.N_snap_part = f['Position'].shape[0]
        return ((f.to_mesh(self.N_snap,
                           dtype=self.pdtype).to_real_field(normalize=False) *
                 (self.N_snap**3 / self.N_snap_part) - 1.0) *
                (1.5 * (self.cosmo_paras['h'] * 100)**2 *
                 self.cosmo_paras['Omega_m'] / a))

    def read_snap_velocity(self, path, snap_type, a):
        if (snap_type == 'CSV'):
            # Start reading initial snapshot and setting initial data
            f = CSVCatalog(path, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi'],
                           dtype=self.pdtype)

            f['Position'] = (f['x'][:, None]) * [1/self.hL_unit, 0, 0] \
                + (f['y'][:, None]) * [0, 1/self.hL_unit, 0] \
                + (f['z'][:, None]) * [0, 0, 1/self.hL_unit]

            f.attrs['BoxSize'] = self.L_snap
        elif (snap_type == 'Gadget1'):
            f = Gadget1Catalog(path,
                               columndefs=[
                                   (
                                       'Position',
                                       ('auto', 3),
                                       'all',
                                   ),
                                   (
                                       'GadgetVelocity',
                                       ('auto', 3),
                                       'all',
                                   ),
                                   (
                                       'ID',
                                       'auto',
                                       'all',
                                   ),
                               ])
            f['Position'] *= 1 / self.hL_unit
            f.attrs['BoxSize'] = self.L_snap
        else:
            raise ValueError('Snap data type is not supported!')

        self.N_snap_part = f['Position'].shape[0]
        f['vx'] = f['GadgetVelocity'][:, 0] \
            * (npy.sqrt(a)) / (3e5)
        f['vy'] = f['GadgetVelocity'][:, 1] \
            * (npy.sqrt(a)) / (3e5)
        f['vz'] = f['GadgetVelocity'][:, 2] \
            * (npy.sqrt(a)) / (3e5)
        return (f.to_mesh(self.N_snap, value='vx',
                          dtype=self.pdtype).to_real_field(normalize=True),
                f.to_mesh(self.N_snap, value='vy',
                          dtype=self.pdtype).to_real_field(normalize=True),
                f.to_mesh(self.N_snap, value='vz',
                          dtype=self.pdtype).to_real_field(normalize=True))

    def Omega_gen(self, delta, vx_un, vy_un, vz_un, Hubble, a, r):
        pp = npy.zeros((self.N_snap, self.N_snap, self.N_snap, 3),
                       dtype=self.pdtype)
        pp[:, :, :, 0] = vx_un
        pp[:, :, :, 1] = vy_un
        pp[:, :, :, 2] = vz_un
        x_list = npy.array([
            r * npy.sin(self.theta_list) * npy.cos(self.phi_list) +
            self.origin[0],
            r * npy.sin(self.theta_list) * npy.sin(self.phi_list) +
            self.origin[1], r * npy.cos(self.theta_list) + self.origin[2]
        ],
                           dtype=self.pdtype)
        x_list = npy.ascontiguousarray(npy.transpose(x_list))

        Omega = npy.ascontiguousarray(
            ut.inverse_Lap(
                1.5 * (self.cosmo_paras['h']* 100)**2 * self.cosmo_paras['Omega_m'] / a
                    * ut.derv_scalar(pp, self.L_snap, self.N_snap) + \
                       Hubble * (delta), self.L_snap, self.N_snap), dtype=self.pdtype)
        Omega = utC.interp(Omega, self.dx, x_list)
        return Omega

    def Phi_Pi_gen(self, rf, r, theta_list, phi_list, win_func=False):
        x_list = npy.array([
            r * npy.sin(theta_list) * npy.cos(phi_list) + self.origin[0],
            r * npy.sin(theta_list) * npy.sin(phi_list) + self.origin[1],
            r * npy.cos(theta_list) + self.origin[2]
        ],
                           dtype=self.pdtype)
        x_list = npy.ascontiguousarray(npy.transpose(x_list))

        Phi = npy.ascontiguousarray(ut.inverse_Lap(rf, self.L_snap,
                                                   self.N_snap, win_func),
                                    dtype=self.pdtype)
        Pi = utC.f_r_derv(Phi, npy.array([self.L_snap / self.N_snap] * 3), r,
                          x_list)
        Phi = utC.interp(Phi, self.dx, x_list)

        return (Phi, Pi)

    def init_build_lcmetric(self):
        self.init_metric()
        self.build_lcmetric()

    def build_lcmetric(self):
        self.met.build_lcmetric()

    def init_metric(self):
        if hasattr(self, 'depo_method') == True:
            depo_method = self.depo_method
        else:
            raise ValueError('depo_method is not specified!')

        sm_dx = 0
        if hasattr(self, 'smoothing') == True and self.smoothing == True:
            sm_dx = self.L_snap / self.N_snap

        self.met.init_from_slice(self.init_z,
                                 self.init_r,
                                 self.delta,
                                 self.vw,
                                 self.Phi_i,
                                 self.Pi_i,
                                 self.cosmo_paras,
                                 self.final_r,
                                 self.Phi_f,
                                 Omega_i=self.Omega_i,
                                 depo_method=depo_method,
                                 sm_dx=sm_dx)


class LightconeFromConePhi(Lightcone):
    def to_r(self, n, nr):
        return n / nr * (self.init_r - self.final_r) + self.final_r

    def __init__(self,
                 cone_path,
                 origin,
                 cosmo_paras,
                 L_snap,
                 N_snap,
                 init_z,
                 final_z,
                 NR,
                 NSIDE,
                 zel_z=None,
                 Phi_zel_path=None,
                 cone_type='UNFORMATTED',
                 snap_type='Gadget1',
                 **kwargs):
        Lightcone.__init__(self, origin, cosmo_paras, L_snap, N_snap, NR,
                           NSIDE, **kwargs)

        self.init_z = init_z
        self.final_z = final_z
        self.zel_z = zel_z
        self.init_a = 1 / (1 + self.init_z)
        self.final_a = 1 / (1 + self.final_z)

        # Initial comoving distance
        self.init_r = scpy.integrate.quad(self.Hint, 0, init_z)[0]

        # Final comoving distance
        self.final_r = scpy.integrate.quad(self.Hint, 0, final_z)[0]

        self.a = npy.array([
            1 / (1 + scpy.optimize.minimize_scalar(
                self.inverse_Hint, args=(self.to_r(n, self.NR))).x)
            for n in range(NR + 1)
        ])

        if (zel_z is not None):
            self.zel_a = 1 / (1 + zel_z)
            self.zel_r = scpy.integrate.quad(self.Hint, 0, zel_z)[0]

        # print("Finishing reading snaps")
        print("Starting reading lightcone data", flush=True)

        rf_i0 = None
        rf_i1 = None
        rf_i2 = None
        if (zel_z is not None):
            Phi_snap_I = \
                self.read_snap_density(Phi_zel_path, snap_type, self.zel_a)

            Phi_snap_I = 5 * ut.inverse_Lap(Phi_snap_I, self.L_snap,
                                            self.N_snap)

            rf_i0 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 0),
                                          dtype=self.pdtype)
            rf_i1 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 1),
                                          dtype=self.pdtype)
            rf_i2 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 2),
                                          dtype=self.pdtype)

        self.delta, self.vw, counts, self.met.sols['Phi'] = \
            self.read_lc_density_phi(cone_path, cone_type, rf_i0, rf_i1, rf_i2)

        for step in range(self.NR + 1):
            self.met.sols['Phi'][step] *= \
                (1.5 * (self.cosmo_paras['h'] * 100)**2
                 * self.cosmo_paras['Omega_m'] * (self.a[step])) \
                * (self.init_r/(2 * npy.pi))**2 / npy.sqrt(2 * npy.pi)

    def read_lc_density_phi(
            self,
            path,
            cone_type,
            rf_i0=None,
            rf_i1=None,
            rf_i2=None,
            chunk=20000000,
            np=7,
    ):
        delta = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        delta.fill(-1)
        vw = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        counts = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        count_density = 1 / (self.L_snap / self.N_snap)**3 \
            / (self.N_snap**3 / self.N_snap_part)

        Phi = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        if (cone_type == 'UNFORMATTED'):
            files = glob.glob(path)
            self.N_lc_part = 0
            cnt = 0
            pdata = []
            print("Total light-cone files num is " + str(len(files)))
            for i in range(len(files)):
                file = str(files[i])

                cnt += ut.unf_read_file(file, pdata, np)
                if (cnt >= chunk or i == len(files) - 1):
                    pdata = npy.ascontiguousarray(
                        npy.array(pdata).reshape(cnt, np)[:, 0:np], dtype=self.pdtype)\
                        * npy.array([1/self.hL_unit, 1/self.hL_unit, 1/self.hL_unit,
                                     1/3e5, 1/3e5, 1/3e5, 1], dtype=self.pdtype)

                    if (self.zel_z is not None):
                        pdata += \
                            utC.interp(rf_i0, self.dx,
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                            * npy.array([1, 0, 0, 0, 0, 0, 0], dtype=self.pdtype) + \
                            utC.interp(rf_i1, self.dx,
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                            * npy.array([0, 1, 0, 0, 0, 0, 0], dtype=self.pdtype) + \
                            utC.interp(rf_i2, self.dx,
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                            * npy.array([0, 0, 1, 0, 0, 0, 0], dtype=self.pdtype)
                    lc_CIC.deposit_with_wgt(pdata, self.origin, delta,
                                            count_density, vw, counts,
                                            self.init_r, self.final_r, self.NR,
                                            self.NSIDE, Phi)
                    pdata = []
                    self.N_lc_part += cnt
                    cnt = 0
            vw /= counts
            vw = npy.nan_to_num(vw)
            Phi /= counts
            Phi = npy.nan_to_num(Phi)

        elif (cone_type == 'CSV'):
            f = CSVCatalog(path, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi'],
                           dtype=self.pdtype)
            f['Particles'] = \
                (f['x'][:, None]) \
                * npy.array([1 / self.hL_unit, 0, 0, 0, 0, 0, 1]) \
                + (f['y'][:, None]) \
                * npy.array([0, 1 / self.hL_unit, 0, 0, 0, 0, 1],) \
                + (f['z'][:, None]) \
                * npy.array([0, 0, 1 / self.hL_unit, 0, 0, 0, 1],)\
                + (f['vx'][:, None]) * npy.array([0, 0, 0, 1/3e5, 0, 0, 1],) \
                + (f['vy'][:, None]) * npy.array([0, 0, 0, 0, 1/3e5, 0, 1],) \
                + (f['vz'][:, None]) * npy.array([0, 0, 0, 0, 0, 1/3e5, 1],)

            f.attrs['BoxSize'] = self.L_snap
            f['Particles'] = f['Particles'].rechunk(chunk)

            self.N_lc_part = f['Particles'].shape[0]

            for d in f['Particles'].partitions:
                pdata = d.compute()
                if (self.zel_z is not None):
                    pdata += \
                        utC.interp(rf_i0, self.dx,\
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [1,0,0,0,0,0,0] +\
                        utC.interp(rf_i1, self.dx,
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [0,1,0,0,0,0,0] +\
                        utC.interp(rf_i2, self.dx,
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [0,0,1,0,0,0,0]
                    lc_CIC.deposit_with_wgt(pdata, self.origin, delta,
                                            count_density, vw, counts,
                                            self.init_r, self.final_r, self.NR,
                                            self.NSIDE, Phi)
            vw /= counts
            vw = npy.nan_to_num(vw)
            Phi /= counts
            Phi = npy.nan_to_num(Phi)
        else:
            raise ValueError('Cone data type is not supported')

        return (delta, vw, counts, Phi[0:self.NR + 1])


class LightconeFromCone(Lightcone):
    """Generating lightcone mesh from particle cone data
    """
    def to_r(self, n, nr):
        return n / nr * (self.init_r - self.final_r) + self.final_r

    def __init__(self,
                 cone_path,
                 Phi_i_path,
                 Phi_f_path,
                 origin,
                 cosmo_paras,
                 L_snap,
                 N_snap,
                 init_z,
                 final_z,
                 NR,
                 NSIDE,
                 zel_z=None,
                 Phi_zel_path=None,
                 cone_type='CSV',
                 snap_type='Gadget1',
                 lensing_kappa=False,
                 chunk=20000000,
                 **kwargs):
        """Initializing lightcone object from particle cone data.

        Parameters:
        -----
        cone_path: string
        An regular expression of path that match ALL light-cone files

        Phi_i_path: string
        An regular expression of path that match ALL files representing
        the initial slice (high z)

        Phi_f_path: string
        An regular expression of path that match ALL files representing
        the final slice (low z)

        origin: float, shape(3,)

        cosmo_paras: dict
        need to contain keywords 'h', 'Omega_m', 'Omega_L'

        L_snap: float, Box size in h^-1 Mpc

        N_snap: int, Box resolution

        init_z: float, redshift of the initial slice

        final_z: float, redshift of the initial slice

        NR: int, radial resolution of the lightcone mesh

        NSIDE: int, helpix NSIDE, angular resolution

        zel_z: float
        Redshift of the N-body initial data slice, for relativistic
        corrections. It it is None, no correction

        Phi_zel_path: string
        Path for the N-body initial data slice, can be None

        cone_type: string, type of the lightcone particle files, Gadget1 or CSV

        snap_type: string, type of the snapshots, Gadget1 or CSV

        lensing_kappa: bool
        If also deposite particles when reading light-cone data ot calculate
        lensing convergence

        chunk: int, data trunk for reading, set it according to your memory size

        depo_method: string:'NGP'
        mass depositing scheme, 'NGP' or 'CIC'

        linear_Omega: bool: True
        use linear approximation for the initial Omega

        smoothing: bool: False
        whether do the gaussian smoothing
        """
        Lightcone.__init__(self, origin, cosmo_paras, L_snap, N_snap, NR,
                           NSIDE, **kwargs)

        self.init_z = init_z
        self.final_z = final_z
        self.zel_z = zel_z
        self.init_a = 1 / (1 + self.init_z)
        self.final_a = 1 / (1 + self.final_z)

        self.depo_method = kwargs.get('depo_method', 'NGP')
        self.linear_Omega = kwargs.get('linear_Omega', True)
        self.smoothing = kwargs.get('smoothing', False)

        # Initial comoving distance
        self.init_r = scpy.integrate.quad(self.Hint, 0, init_z)[0]

        # Final comoving distance
        self.final_r = scpy.integrate.quad(self.Hint, 0, final_z)[0]

        self.a_shift = npy.array([
            1 / (1 + scpy.optimize.minimize_scalar(
                self.inverse_Hint, args=(self.to_r(n - 1, self.NR))).x)
            for n in range(NR + 3)
        ])

        if (zel_z is not None):
            self.zel_a = 1 / (1 + zel_z)
            self.zel_r = scpy.integrate.quad(self.Hint, 0, zel_z)[0]

        # Initial and final Hubble
        self.Hi = self.H(self.init_z)
        self.Hf = self.H(self.final_z)

        print("Starting reading initial snap", flush=True)
        delta_i = self.read_snap_density(Phi_i_path, snap_type, self.init_a)
        self.Phi_i, self.Pi_i = self.Phi_Pi_gen(delta_i,
                                                self.init_r,
                                                self.theta_list,
                                                self.phi_list,
                                                win_func=True)
        self.Omega_i = None
        if self.linear_Omega == False:
            vx, vy, vz = self.read_snap_velocity(Phi_i_path, snap_type,
                                                 self.init_a)
            Hubble = ut.H(self.init_z, self.cosmo_paras['h'] * 100,
                          self.cosmo_paras['Omega_m'],
                          self.cosmo_paras['Omega_L'])

            self.Omega_i = -self.Omega_gen(delta_i, vx, vy, vz, Hubble,
                                           self.init_a, self.init_r)

        print("Starting reading final snap", flush=True)
        delta_f = self.read_snap_density(Phi_f_path, snap_type, self.final_a)
        self.Phi_f, self.Pi_f = self.Phi_Pi_gen(delta_f,
                                                self.final_r,
                                                self.theta_list,
                                                self.phi_list,
                                                win_func=True)

        # print("Finishing reading snaps")
        print("Starting reading lightcone data", flush=True)

        rf_i0 = None
        rf_i1 = None
        rf_i2 = None

        if (zel_z is not None):
            Phi_snap_I = \
                self.read_snap_density(Phi_zel_path, snap_type, self.zel_a)

            Phi_snap_I = 5 * ut.inverse_Lap(Phi_snap_I, self.L_snap,
                                            self.N_snap)

            rf_i0 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 0),
                                          dtype=self.pdtype)
            rf_i1 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 1),
                                          dtype=self.pdtype)
            rf_i2 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 2),
                                          dtype=self.pdtype)

        # reading light-cone
        self.delta, self.vw, counts = \
            self.read_lc_density(cone_path, cone_type, rf_i0, rf_i1, rf_i2,
                                 lensing_kappa=lensing_kappa, chunk=chunk, np=6)

    # Reading lp-cola unformatted files
    def unf_read_file(self, file, p_list=[], np=7):
        with open(file, mode="rb") as f:
            tot_n = 0
            cnt = 0
            while (True):
                cnt += 1
                r = f.read(4)
                if not r: break

                a1 = struct.unpack('i', r)

                r = f.read(a1[0])
                n = struct.unpack('i', r)

                r = f.read(8)
                a, b = struct.unpack('2i', r)

                r = f.read(b)
                p_list.extend(struct.unpack(str(n[0] * np) + 'f', r))

                r = f.read(4)
                tot_n += n[0]
        f.close()
        return tot_n

    def read_lc_density(self,
                        path,
                        cone_type,
                        rf_i0=None,
                        rf_i1=None,
                        rf_i2=None,
                        chunk=20000000,
                        np=7,
                        lensing_kappa=False):
        delta = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        delta.fill(-1)
        vw = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        counts = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        count_density = 1 / (self.L_snap / self.N_snap)**3 \
            / (self.N_snap**3 / self.N_snap_part)

        if (lensing_kappa is True):
            self.kappa1 = npy.zeros((self.NR + 2, self.NPIX),
                                    dtype=self.pdtype)
            self.kappa2 = npy.zeros((self.NR + 2, self.NPIX),
                                    dtype=self.pdtype)

        if (cone_type == 'UNFORMATTED'):
            files = glob.glob(path)
            self.N_lc_part = 0
            cnt = 0
            pdata = []
            print("Total light-cone files num is " + str(len(files)))
            for i in range(len(files)):
                file = files[i]
                cnt += self.unf_read_file(file, pdata, np)
                if (cnt >= chunk or i == len(files) - 1):
                    pdata = npy.ascontiguousarray(
                        npy.array(pdata).reshape(cnt, np)[:, 0:6], dtype=self.pdtype)\
                        * npy.array([1/self.hL_unit, 1/self.hL_unit, 1/self.hL_unit,
                                     1/3e5, 1/3e5, 1/3e5], dtype=self.pdtype)

                    if (self.zel_z is not None):
                        pdata += \
                            utC.interp(rf_i0, self.dx,\
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                       * [1,0,0,0,0,0] + \
                            utC.interp(rf_i1, self.dx,
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                       * [0,1,0,0,0,0] + \
                            utC.interp(rf_i2, self.dx,
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                       * [0,0,1,0,0,0]
                    lc_CIC.deposit(pdata, self.origin, delta, count_density, \
                                   vw, counts, self.init_r, self.final_r, self.NR,
                                   self.NSIDE, self.depo_method)
                    if (lensing_kappa is True):
                        lc_CIC.lensing_kappa_deposit(pdata, self.a_shift,
                                                     self.origin, self.kappa1,
                                                     self.kappa2, self.init_r,
                                                     self.final_r, self.NR,
                                                     self.NSIDE)
                    pdata = []
                    self.N_lc_part += cnt
                    cnt = 0
            vw /= counts
            vw = npy.nan_to_num(vw)
        elif (cone_type == 'CSV'):
            f = CSVCatalog(path, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi'],
                           dtype=self.pdtype)
            f['Particles'] = \
                (f['x'][:, None]) \
                * npy.array([1 / self.hL_unit, 0, 0, 0, 0, 0]) \
                + (f['y'][:, None]) \
                * npy.array([0, 1 / self.hL_unit, 0, 0, 0, 0],) \
                + (f['z'][:, None]) \
                * npy.array([0, 0, 1 / self.hL_unit, 0, 0, 0],)\
                + (f['vx'][:, None]) * npy.array([0, 0, 0, 1/3e5, 0, 0],) \
                + (f['vy'][:, None]) * npy.array([0, 0, 0, 0, 1/3e5, 0],) \
                + (f['vz'][:, None]) * npy.array([0, 0, 0, 0, 0, 1/3e5],)

            f.attrs['BoxSize'] = self.L_snap
            f['Particles'] = f['Particles'].rechunk(chunk)

            self.N_lc_part = f['Particles'].shape[0]

            for d in f['Particles'].partitions:
                pdata = d.compute()
                if (self.zel_z is not None):
                    pdata += \
                        utC.interp(rf_i0, self.dx,\
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [1,0,0,0,0,0] +\
                        utC.interp(rf_i1, self.dx,
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [0,1,0,0,0,0] +\
                        utC.interp(rf_i2, self.dx,
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [0,0,1,0,0,0]
                    lc_CIC.deposit(pdata, self.origin, delta, count_density, \
                                   vw, counts, self.init_r, self.final_r, self.NR,
                                   self.NSIDE, self.depo_method)
                    if (lensing_kappa is True):
                        lc_CIC.lensing_kappa_deposit(pdata, self.a_shift,
                                                     self.origin, self.kappa1,
                                                     self.kappa2, self.init_r,
                                                     self.final_r, self.NR,
                                                     self.NSIDE)
            vw /= counts
            vw = npy.nan_to_num(vw)
        else:
            raise ValueError('Cone data type is not supported')

        return (delta, vw, counts)


class LightconeFromSnaps(Lightcone):
    """Lightcone mesh from snapshots, can choose to use relaxation
    or not.
    """
    def to_r(self, n, nr):
        return n / nr * (self.init_r - self.final_r) + self.final_r

    def __init__(self,
                 snaps_path,
                 n_threads,
                 origin,
                 cosmo_paras,
                 L_snap,
                 N_snap,
                 init_snap_i,
                 final_snap_i,
                 NR,
                 NSIDE,
                 zel_z=None,
                 zel_path=None,
                 snap_type='Gadget1',
                 NR_is_N_snap=False,
                 lensing_kappa=False,
                 need_reduce=False,
                 reduce_ratio=4,
                 chunk=2**62,
                 **kwargs):
        """Initializing lightcone object from particle cone data.

        Parameters:
        -----
        snap_path: string
        An regular expression of path that match ALL snapshots files.
        The name for each snap should be followed by a '.thread',
        where thread represents the idx of thread generating this snap.

        n_threads: threads number of n-body code

        origin: float, shape(3,)

        cosmo_paras: dict
        need to contain keywords 'h', 'Omega_m', 'Omega_L'

        L_snap: float, Box size in h^-1 Mpc

        N_snap: int, Box resolution

        init_snap_i: int, index of the initial snap you want to push into the mesh

        final_snap_i: int,index of the final snap you want to push into the mesh

        NR: int, radial resolution of the lightcone mesh

        NSIDE: int, helpix NSIDE, angular resolution

        zel_z: float
        Redshift of the N-body initial data slice, for relativistic
        corrections. It it is None, no correction

        Phi_zel_path: string
        Path for the N-body initial data slice, can be None

        snap_type: string, type of the snapshots, Gadget1 or CSV

        NR_is_N_snap: bool, optional
        If NR should just be N_snap, so that a very undersampled
        light-cone mesh will be built representing a traditional treatment

        lensing_kappa: bool, optional
        If also deposite particles when reading light-cone data ot calculate
        lensing convergence

        need_reduce: bool, optional
        If the number of snaps should be reduced by a factor of 4
        (to gnerate even more sparse data)
        """

        Lightcone.__init__(self, origin, cosmo_paras, L_snap, N_snap, NR,
                           NSIDE, **kwargs)

        self.zel_z = zel_z

        self.depo_method = kwargs.get('depo_method', 'NGP')
        self.linear_Omega = kwargs.get('linear_Omega', True)
        self.smoothing = kwargs.get('smoothing', False)
        self.lc_shift = kwargs.get('lc_shift', 0)

        if (zel_z is not None):
            self.zel_a = 1 / (1 + zel_z)
            self.zel_r = scpy.integrate.quad(self.Hint, 0,
                                             zel_z)[0] + self.lc_shift

        t = sorted(glob.glob(snaps_path))
        self.names = list()
        files = list()
        import os
        if (n_threads == 1):
            for i in range(0, len(t), n_threads):
                self.names.append(os.path.commonprefix(t[i:i + 1]) + '*')
        else:
            for i in range(0, len(t), n_threads):
                self.names.append(
                    os.path.commonprefix(t[i:i + n_threads]) + '[0-9]*')

        self.n_snaps = len(self.names)

        for file in self.names:
            files.append(Gadget1Catalog(file,columndefs=[ \
                ('Position', ('auto', 3), 'all',),
                ('GadgetVelocity',  ('auto', 3), 'all', ),
                ('ID', 'auto', 'all', )]))

        print("Initial snap is set at redshift " +
              str(files[init_snap_i].attrs['Redshift']),
              flush=True)

        print("Final snap is set at redshift " +
              str(files[final_snap_i].attrs['Redshift']),
              flush=True)

        self.init_z = files[init_snap_i].attrs['Redshift']
        self.final_z = files[final_snap_i].attrs['Redshift']
        self.init_a = 1 / (1 + self.init_z)
        self.final_a = 1 / (1 + self.final_z)

        # Initial comoving distance
        self.init_r = scpy.integrate.quad(self.Hint, 0,
                                          self.init_z)[0] + self.lc_shift

        # Final comoving distance
        self.final_r = scpy.integrate.quad(self.Hint, 0,
                                           self.final_z)[0] + self.lc_shift

        if (need_reduce is True):
            temp = self.names[1::reduce_ratio]
            temp.insert(0, self.names[0])
            self.names = temp

            self.n_snaps = len(self.names)
            # clean-up original files array
            files = list()
            for file in self.names:
                files.append(Gadget1Catalog(file,columndefs=[ \
                    ('Position', ('auto', 3), 'all',),
                    ('GadgetVelocity',  ('auto', 3), 'all', ),
                    ('ID', 'auto', 'all', )]))

        # Check if the input data are in increasing order
        for fi in range(1, self.n_snaps):
            if (files[fi].attrs['Redshift'] <=
                    files[fi - 1].attrs['Redshift']):
                raise ValueError(
                    'The redshifts are not increasing for input snaps!')

        if (NR_is_N_snap is True):
            """If we do want to build a lightcone mesh just based on snapshots
            data without any further information.
            """
            #self.NR = init_snap_i - final_snap_i
            self.NR = self.n_snaps - 2
            print('Setting the radial resolution NR as ' + str(self.NR) +
                  ' since we do want NR is N_snap',
                  flush=True)

            self.a = npy.array([1/(1+scpy.optimize.minimize_scalar(
                self.inverse_Hint, args=(self.to_r(n, self.NR))).x) \
                              for n in range(self.NR+1)])
            self.Phi = npy.zeros((self.NR + 1, self.NPIX), dtype=self.pdtype)
            self.Pi = npy.zeros((self.NR + 1, self.NPIX), dtype=self.pdtype)
            self.dPhi = npy.zeros((self.NR + 1, self.NPIX), dtype=self.pdtype)
            self.Omega = npy.zeros((self.NR + 1, self.NPIX), dtype=self.pdtype)

            self.r_list = npy.zeros(self.NR + 1)
            self.z_list = npy.zeros(self.NR + 1)
            #for fi in range(init_snap_i, final_snap_i - 1, -1):
            for fi in range(self.n_snaps - 1, 0, -1):
                ni = self.NR - (self.n_snaps - 1 - fi)

                files[fi]['Position'] *= 1 / self.hL_unit
                files[fi].attrs['BoxSize'] = self.L_snap
                r = scpy.integrate.quad(
                    self.Hint, 0,
                    files[fi].attrs['Redshift'])[0] + self.lc_shift

                self.r_list[ni] = r
                self.z_list[ni] = files[fi].attrs['Redshift']

                rf = (files[fi].to_mesh(self.N_snap, dtype=self.pdtype).to_real_field(normalize=False)  \
                      * ( self.N_snap**3 / files[fi]['Position'].shape[0] ) - 1.0 )

                # Getting (\delta) * v
                files[fi]['vx'] = files[fi]['GadgetVelocity'][:, 0] \
                    * (npy.sqrt(1 / (1 + files[fi].attrs['Redshift'])) / (3e5))
                files[fi]['vy'] = files[fi]['GadgetVelocity'][:, 1] \
                    * (npy.sqrt(1 / (1 + files[fi].attrs['Redshift'])) / (3e5))
                files[fi]['vz'] = files[fi]['GadgetVelocity'][:, 2] \
                    * (npy.sqrt(1 / (1 + files[fi].attrs['Redshift'])) / (3e5))

                pp = npy.zeros((self.N_snap, self.N_snap, self.N_snap, 3),
                               dtype=self.pdtype)
                pp[:,:,:, 0] = \
                    files[fi].to_mesh(N_snap, value='vx',
                              dtype=self.pdtype).to_real_field(normalize=True)
                pp[:,:,:, 1] = \
                    files[fi].to_mesh(N_snap, value='vy',
                              dtype=self.pdtype).to_real_field(normalize=True)
                pp[:,:,:, 2] = \
                    files[fi].to_mesh(N_snap, value='vz',
                              dtype=self.pdtype).to_real_field(normalize=True)
                Hubble = ut.H(files[fi].attrs['Redshift'],
                              cosmo_paras['h'] * 100, cosmo_paras['Omega_m'],
                              cosmo_paras['Omega_L'])

                self.Omega[ni] = self.Phi_Pi_gen(
                    1.5 * (cosmo_paras['h']* 100)**2 * cosmo_paras['Omega_m']
                    * (1 + files[fi].attrs['Redshift'])\
                    * (ut.derv_scalar(pp, self.L_snap, self.N_snap) + \
                       Hubble * (rf )), r, self.theta_list, self.phi_list)[0]

                rf *= (1.5 * (cosmo_paras['h'] * 100)**2 *
                       cosmo_paras['Omega_m'] *
                       (1 + files[fi].attrs['Redshift']))
                snap = npy.ascontiguousarray(ut.inverse_Lap(rf,
                                                            self.L_snap,
                                                            self.N_snap,
                                                            win_func=True),
                                             dtype=self.pdtype)
                print(str(ni) + ' ' + str(r) + ' ' +
                      str(files[fi].attrs['Redshift']),
                      flush=True)

                self.Phi[ni] = utC.interp(snap, npy.array(rf.BoxSize/rf.Nmesh),
                    npy.ascontiguousarray([ [\
                    r * npy.sin(self.theta_list[i]) * npy.cos(self.phi_list[i]) + origin[0],\
                    r * npy.sin(self.theta_list[i]) * npy.sin(self.phi_list[i]) + origin[1],\
                    r * npy.cos(self.theta_list[i]) + origin[2]]  \
                                            for i in range(self.NPIX)], dtype=self.pdtype))

                if (fi == self.n_snaps - 1):
                    pre_snap = snap.copy()

                self.dPhi[ni] = \
                        utC.interp(
                            snap - pre_snap, npy.array(rf.BoxSize / rf.Nmesh),
                            npy.ascontiguousarray(
                        [ [
                            r * npy.sin(self.theta_list[i]) * npy.cos(self.phi_list[i]) + origin[0],\
                            r * npy.sin(self.theta_list[i]) * npy.sin(self.phi_list[i]) + origin[1],\
                            r * npy.cos(self.theta_list[i]) + origin[2]]  \
                          for i in range(self.NPIX)], dtype=self.pdtype))
                pre_snap = snap.copy()

                self.Pi[ni] = utC.f_r_derv(
                    snap, npy.array(rf.BoxSize / rf.Nmesh), r,
                    npy.ascontiguousarray(
                        [ [
                            r * npy.sin(self.theta_list[i]) * npy.cos(self.phi_list[i]) + origin[0],\
                            r * npy.sin(self.theta_list[i]) * npy.sin(self.phi_list[i]) + origin[1],\
                            r * npy.cos(self.theta_list[i]) + origin[2]]  \
                          for i in range(self.NPIX)], dtype=self.pdtype))

                #del files[fi]
            self.met.sols['Phi'] = self.Phi
            self.met.sols['Pi'] = self.Pi
            return

        # shifted (-1) version of the scale factor,
        # will also cover the region that one grid outside
        # (final_r, init_r)
        self.a_shift = npy.ascontiguousarray([1 / (1+scpy.optimize.minimize_scalar(
            self.inverse_Hint, args=(self.to_r(n - 1, self.NR))).x) \
                                              for n in range(NR+3)])

        print("No. of snaps is " + str(self.n_snaps) +
              ", and No. of particles in each snap is " +
              str(files[0]['Position'].shape[0]),
              flush=True)

        print("Starting reading initial snap", flush=True)
        delta_i = self.read_snap_density(self.names[init_snap_i], snap_type,
                                         self.init_a)
        self.Phi_i, self.Pi_i = self.Phi_Pi_gen(delta_i,
                                                self.init_r,
                                                self.theta_list,
                                                self.phi_list,
                                                win_func=True)

        self.Omega_i = None
        if self.linear_Omega == False:
            vx, vy, vz = self.read_snap_velocity(self.names[init_snap_i],
                                                 snap_type, self.init_a)
            Hubble = ut.H(self.init_z, self.cosmo_paras['h'] * 100,
                          self.cosmo_paras['Omega_m'],
                          self.cosmo_paras['Omega_L'])
            self.Omega_i = -self.Omega_gen(delta_i, vx, vy, vz, Hubble,
                                           self.init_a, self.init_r)

        print("Starting reading final snap", flush=True)
        self.Phi_f, self.Pi_f = self.Phi_Pi_gen(self.read_snap_density(
            self.names[final_snap_i], snap_type, self.final_a),
                                                self.final_r,
                                                self.theta_list,
                                                self.phi_list,
                                                win_func=True)

        rf_i0 = None
        rf_i1 = None
        rf_i2 = None

        if (zel_z is not None):
            Phi_snap_I = self.read_snap_density(zel_path, snap_type,
                                                self.zel_a)

            Phi_snap_I = 5 * ut.inverse_Lap(Phi_snap_I, self.L_snap,
                                            self.N_snap)
            rf_i0 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 0),
                                          dtype=self.pdtype)
            rf_i1 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 1),
                                          dtype=self.pdtype)
            rf_i2 = npy.ascontiguousarray(ut.inverse_derv(
                Phi_snap_I, L_snap, N_snap, 2),
                                          dtype=self.pdtype)

        self.N_snap_part = files[0]['Position'].shape[0]

        self.snap_den = dens.DensFromSnaps(self.origin, self.L_snap,
                                           files[0]['Position'].shape[0],
                                           self.init_r, self.cosmo_paras,
                                           self.hL_unit)

        self.delta = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        self.delta.fill(-1)
        self.vw = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        counts = npy.zeros((self.NR + 2, self.NPIX), dtype=self.pdtype)
        count_density = 1 / (self.L_snap / self.N_snap)**3 \
            / (self.N_snap**3 / self.N_snap_part)
        self.tot_p_num = 0

        if (lensing_kappa is True):
            self.kappa1 = npy.zeros((self.NR + 2, self.NPIX),
                                    dtype=self.pdtype)
            self.kappa2 = npy.zeros((self.NR + 2, self.NPIX),
                                    dtype=self.pdtype)

        ##########Reading slice by slice##############
        for fi in range(self.n_snaps - 1, -1, -1):
            #if (files[fi].attrs['Redshift'] < 1e-3):
            #    continue
            tau = -scpy.integrate.quad(self.Hint, 0,
                                       files[fi].attrs['Redshift'])[0]
            if (fi < self.n_snaps - 1):
                dtau = scpy.integrate.quad(
                    self.Hint, 0, files[fi + 1].attrs['Redshift'])[0] + tau
            else:
                dtau = 0

            z = files[fi].attrs['Redshift']
            a = 1 / (1 + z)

            # Rechunking data
            files[fi]['Position'] = files[fi]['Position'].rechunk(chunk)
            files[fi]['GadgetVelocity'] = files[fi]['GadgetVelocity'].rechunk(
                chunk)
            files[fi]['ID'] = files[fi]['ID'].rechunk(chunk)

            for p, v, i, in zip(files[fi]['Position'].partitions,
                                files[fi]['GadgetVelocity'].partitions,
                                files[fi]['ID'].partitions):
                pos = npy.ascontiguousarray(p / self.hL_unit,
                                            dtype=self.pdtype)
                vel = npy.ascontiguousarray(v * (npy.sqrt(a) / (3e5)),
                                            dtype=self.pdtype)
                IDs = npy.ascontiguousarray(i)

                if (self.zel_z is not None):
                    pos += \
                        utC.interp(rf_i0, self.dx,
                                   npy.ascontiguousarray(pos[:, 0:3]))[:, None] \
                                   * [1, 0, 0] +\
                        utC.interp(rf_i1, self.dx,
                                   npy.ascontiguousarray(pos[:, 0:3]))[:, None] \
                                   * [0, 1, 0] +\
                        utC.interp(rf_i2, self.dx,
                                   npy.ascontiguousarray(pos[:, 0:3]))[:, None] \
                                   * [0, 0, 1]

                state = self.snap_den.proc_snap_chunk(pos, vel, IDs,
                                                      tau - self.lc_shift,
                                                      dtau)
                if (fi == self.n_snaps - 1):
                    continue

                if (state is False):
                    raise ValueError('Reading snapshot No. ' + str(fi) +
                                     ' failed!')

                p_num = int(self.snap_den.p_num() / 6)
                self.tot_p_num += p_num
                print(str(fi) + ' ' + str(files[fi].attrs['Redshift']) + ' ' +
                      str(dtau) + ' ' + str(p_num),
                      flush=True)
                if (p_num == 0):
                    continue
                pdata = npy.array(self.snap_den.get_pdata()).reshape(p_num, 6)

                lc_CIC.deposit(pdata, self.origin, self.delta, count_density,
                               self.vw, counts, self.init_r, self.final_r,
                               self.NR, self.NSIDE, self.depo_method)
                if (lensing_kappa is True):
                    lc_CIC.lensing_kappa_deposit(pdata, self.a_shift,
                                                 self.origin, self.kappa1,
                                                 self.kappa2, self.init_r,
                                                 self.final_r, self.NR,
                                                 self.NSIDE)

                self.snap_den.clear_lc()

            # Change the flag to non-first snap
            self.snap_den.not_first_snap()

        self.vw /= counts
        self.vw = npy.nan_to_num(self.vw)

        del files
        # self.met.init_from_slice(self.init_z, self.init_r, self.delta, self.vw,
        #                          self.Phi_i, self.Pi_i, self.cosmo_paras,
        #                          self.final_r, self.Phi_f)
        del self.snap_den
