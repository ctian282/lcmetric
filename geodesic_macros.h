
#ifndef GEODESIC_MACROS
#define GEODESIC_MACROS

/*
 * applying functions to lots of vars
 */

#define PW2(x) ( (x) * (x))



#define GEODESIC_APPLY_TO_FIELDS_ARGS(function, ...)    \
  function(theta, __VA_ARGS__);                         \
  function(phi, __VA_ARGS__);                           \
  function(k0, __VA_ARGS__);                            \
  function(dDAdt, __VA_ARGS__);                         \
  function(DA, __VA_ARGS__);                            \
  function(nr, __VA_ARGS__);                            \
  function(ntheta, __VA_ARGS__);                        \
  function(nphi, __VA_ARGS__);                          \
  function(e1r, __VA_ARGS__);                           \
  function(e1theta, __VA_ARGS__);                       \
  function(e1phi, __VA_ARGS__);                         \
  function(e2r, __VA_ARGS__);                           \
  function(e2theta, __VA_ARGS__);                       \
  function(e2phi, __VA_ARGS__);                         \
  function(beta1, __VA_ARGS__);                         \
  function(beta2, __VA_ARGS__);                         

#define GEODESIC_APPLY_TO_FIELDS(function)      \
  function(theta);                              \
  function(phi);                                \
  function(k0);                                 \
  function(dDAdt);                              \
  function(DA);                                 \
  function(nr);                                 \
  function(ntheta);                             \
  function(nphi);                               \
  function(e1r);                                \
  function(e1theta);                            \
  function(e1phi);                              \
  function(e2r);                                \
  function(e2theta);                            \
  function(e2phi);                              \
  function(beta1);                              \
  function(beta2);                              

#define GEODESIC_APPLY_TO_TEST_FIELDS(function)      \
  function(theta);                              

#define GEODESIC_APPLY_TO_DERIVED_FIELDS_ARGS(function, ...)    \
  function(z, __VA_ARGS__);                         \

#define GEODESIC_APPLY_TO_DERIVED_FIELDS(function)      \
  function(z);                                          \


#define GEODESIC_APPLY_TO_COMPLEX_FIELDS_ARGS(function, ...)    \
  function(sigma, __VA_ARGS__);                         \
  function(epsilon, __VA_ARGS__);                           \
  function(omega, __VA_ARGS__);                            \

#define GEODESIC_APPLY_TO_COMPLEX_FIELDS(function)      \
  function(sigma);                                      \
  function(epsilon);                                    \
  function(omega);                                 


#define DECLARE_REAL_T(name) \
  real_t name

#define DECLARE_COMPLEX_T(name) \
  std::complex<real_t> name


#define RK2_FIELDS_ALL_CREATE(field)    \
  real_t * field##_a;                   \
  real_t * field##_f;                   \
  real_t * field##_c;

#define RK2_FIELDS_ALL_DEL(field)    \
  delete [] field##_a;                   \
  delete [] field##_f;                   \
  delete [] field##_c;


#define RK2_COMPLEX_FIELDS_ALL_CREATE(field)    \
  std::complex<real_t> * field##_a;             \
  std::complex<real_t> * field##_f;             \
  std::complex<real_t> * field##_c;


#define RK2_FIELDS_ALL_INIT(field, size)        \
  field##_a = new real_t[size]();               \
  field##_f = new real_t[size]();               \
  field##_c = new real_t[size]();


#define RK2_COMPLEX_FIELDS_ALL_INIT(field, size)        \
  field##_a = new std::complex<real_t>[size]();         \
  field##_f = new std::complex<real_t>[size]();         \
  field##_c = new std::complex<real_t>[size]();

#define RK2_ADVANCE_ALL_K1(field)                  \
  field##_c[i] = d##field##_dt(p);              \
  field##_a[i] += field##_c[i] * dtau;

#define RK2_ADVANCE_ALL_K2(field)                  \
  field##_c[i] += d##field##_dt(p);              \
  field##_f[i] += 0.5 * field##_c[i] * dtau;     \
  field##_a[i] = field##_f[i];


#define SET_LOCAL_VALUES(name) \
  p.name = name##_a[p.pid]

#endif
