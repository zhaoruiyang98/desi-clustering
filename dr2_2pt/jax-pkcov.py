def compute_theory_for_covariance_mesh2_spectrum(output_fn, spectrum_fns, window_fn, klim=(0., 0.3)):
    import lsstypes as types
    from jaxpower import (ParticleField, MeshAttrs, compute_spectrum2_covariance)
    mean = types.mean([types.read(fn) for fn in spectrum_fns])
    window = types.read(window_fn)

    mattrs = MeshAttrs(**{name: mean.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    mean_with_sn = mean.map(lambda pole: pole.clone(value=pole.value() + pole.values('shotnoise')))
    # Include sn for compute_spectrum2_covariance for cubic box
    covariance = compute_spectrum2_covariance(mattrs, mean_with_sn)

    sl = slice(0, None, 5)  # rebin to dk = 0.001 h/Mpc
    oklim = (0., 0.35)  # fitted k-range, no need to go to higher k
    smooth = mean.map(lambda pole: pole.clone(k=pole.coords('k', center='mid_if_edges'))).select(k=klim)
    mean = mean.select(k=sl).select(k=oklim)
    window = window.at.observable.select(k=sl).at.observable.select(k=oklim).at.theory.select(k=(0., 1.1 * oklim[1]))
    covariance = covariance.at.observable.select(k=sl).at.observable.select(k=oklim)

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=window.theory.get(ells=0).z)
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, prior_basis=None)
    observable = TracerPowerSpectrumMultipolesObservable(data=mean, wmatrix=window, theory=theory, covariance=covariance)
    likelihood = ObservablesGaussianLikelihood(observable)

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    print(profiles.to_stats(tablefmt='pretty'))
    theory.init.update(k=smooth.get(0).coords('k'))
    params = profiles.bestfit.choice(index='argmax', input=True)
    params = {name: float(value) for name, value in params.items()}
    poles = theory(**params)
    smooth = smooth.clone(value=poles.ravel())
    for pole in smooth:
        pole._meta['z'] = template.z
    smooth.attrs['b1'] = params['b1']
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        smooth.write(output_fn)
    return smooth


def compute_jaxpower_covariance_mesh2_spectrum(output_fn, get_data, get_randoms, get_theory, get_spectrum):
    import jax
    from jaxpower import (ParticleField, get_mesh_attrs, MeshAttrs, compute_fkp2_covariance_window, compute_spectrum2_covariance, interpolate_window_function, read)
    theory = get_theory()
    spectrum = get_spectrum()
    data, randoms = get_data(), get_randoms()
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    data = ParticleField(data[0], data[1][0], attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='jax')
    fftlog = False
    kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
    windows = compute_fkp2_covariance_window(randoms, alpha=data.sum() / randoms.sum(),
                                             interlacing=3, resampler='tsc', los='local', **kw)
    if output_fn is not None and jax.process_index() == 0:
        window = ObservableTree(windows, names=['WW', 'WS', 'SS'])
        fn = Path(output_fn)
        fn = fn.parent / 'covariance_windows' / f'window_{fn.name}'
        logger.info(f'Writing to {fn}')
        window.write(fn)

    if fftlog:
        coords = np.logspace(-2, 8, 8 * 1024)
        windows = [interpolate_window_function(window, coords=coords) for window in windows]

    # delta is the maximum abs(k1 - k2) where the covariance will be computed (to speed up calculation)
    covs_analytical = compute_spectrum2_covariance(windows, get_theory(), flags=['smooth'] + (['fftlog'] if fftlog else []), delta=0.4)

    # Sum all contributions (WW, WS, SS), with W = standard window (multiplying delta), S = shotnoise
    # Here we assumed randoms have a negligible contribution to the shot noise in the measurements
    cov = covs_analytical[0].clone(value=sum(cov.value() for cov in covs_analytical))
    cov = cov.at.observable.match(spectrum)
    cov = cov.clone(observable=spectrum)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        cov.write(output_fn)
    return cov


def get_post_recon_spectrum(cosmo, k=None, z=1., b1=1., smoothing_radius=15., ells=(0, 2, 4), fields=('post', 'post'), shotnoise=0.):
    from scipy import special, integrate
    from cosmoprimo import PowerSpectrumBAOFilter

    if k is None:
        k = np.linspace(0.01, 0.2, 100)

    def weights_leggauss(nx, sym=True):
        """Return weights for Gauss-Legendre integration."""
        x, wx = np.polynomial.legendre.leggauss((1 + sym) * nx)
        if sym:
            x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
        return x, wx

    mu, wmu = weights_leggauss(8)
    q = cosmo.rs_drag
    klin = np.logspace(-3., 2., 1000)
    pklin = cosmo.get_fourier().pk_interpolator(of='delta_cb').to_1d(z=z)
    pknow = PowerSpectrumBAOFilter(pklin, engine='wallish2018').smooth_pk_interpolator()
    k = k[:, None]
    wiggles = pklin(k) - pknow(k)
    f = cosmo.growth_rate(z)
    wmu = np.array([wmu * (2 * ell + 1) * special.legendre(ell)(mu) for ell in ells])
    if fields == ('post', 'post'):
        j0 = special.jn(0, q * klin)
        sk = np.exp(-1. / 2. * (klin * smoothing_radius)**2)
        skc = 1. - sk
        sigma = 1. / (3. * np.pi**2) * integrate.simpson((1. - j0) * skc**2 * pklin(klin), klin)
        ksq = (1 + f * (f + 2) * mu**2) * k**2
        resummed_wiggles = (b1 + f * mu**2)**2 * np.exp(-1. / 2. * ksq * sigma) * wiggles
        pkmu = (b1 + f * mu**2)**2 * pknow(k) + resummed_wiggles + shotnoise
    elif fields == ('pre', 'post'):
        sk = np.exp(-1. / 2. * (klin * smoothing_radius)**2)
        sigma = 1. / (6. * np.pi**2) * integrate.simpson(sk * pklin(klin), klin)
        ksq = (1 + (1 + f)**2 * mu**2) * k**2
        pkmu = np.exp(- 1. / 2. * ksq * sigma) * ((b1 + f * mu**2)**2 * pklin(k) + shotnoise)
    poles = np.sum(pkmu * wmu[:, None, :], axis=-1)
    return poles


def compute_jaxpower_covariance_mesh2_spectrum_pre_correlation_post(output_fn, get_theory, get_spectrum, get_RR, tracer='LRG'):
    from jaxpower import compute_spectrum2_covariance

    fn = Path(output_fn)
    fn = fn.parent / 'covariance_windows' / f'window_{fn.name.replace("spectrum-pre_correlation-post", "spectrum")}'
    window = types.read(fn)

    #kslice = slice(0, None, 10)
    pre_pre = get_theory()#.select(k=kslice)
    smoothing_radius = {'QSO': 30.}.get(tracer[:3], 15.)
    kw = dict(b1=pre_pre.attrs['b1'], z=pre_pre.get(ells=0).z, k=pre_pre.get(ells=0).coords('k'), ells=pre_pre.ells, smoothing_radius=smoothing_radius)
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    spectrum_pre = get_spectrum()#.select(k=kslice)
    post_post = get_post_recon_spectrum(cosmo, fields=('post', 'post'), **kw)  # shotnoise accounted for with window
    post_post = pre_pre.clone(value=post_post.ravel())
    pre_post = get_post_recon_spectrum(cosmo, fields=('pre', 'post'), shotnoise=spectrum_pre.get(0).values('shotnoise').mean(), **kw)  # damped shotnoise
    pre_post = pre_pre.clone(value=pre_post.ravel())
    theory = ObservableTree([pre_pre, pre_post, post_post], fields=[('pre', 'pre'), ('pre', 'post'), ('post', 'post')])

    fields = [('pre',) * 4, ('post',) * 4, ('pre', 'post') * 2, ('post', 'pre') * 2]
    WW = ObservableTree([window.get('WW').get(fields=(0, 0, 0, 0))] * len(fields), fields=fields)
    WS = ObservableTree([window.get('WS').get(fields=(0, 0, 0, 0))] * 2, fields=fields[:2])
    SS = ObservableTree([window.get('SS').get(fields=(0, 0, 0, 0))], fields=fields[:1])  # let's add it back by hand
    windows = WW, WS, SS

    RR = get_RR()
    from lsstypes.types import compute_RR2_window
    slim = (50., 150.)
    RR = RR.clone(counts=RR.values('counts'), norm=RR.attrs['norm_fkp']).select(s=slice(0, None, 4)).select(s=slim)
    sedges = RR.edges('s')
    s = RR.coords('s')
    window_RR = compute_RR2_window(RR, edges=sedges, ells=kw['ells'], ellsin=kw['ells'], kind='RR', resolution=1)
    #print(window_RR.value()[0, 0])
    #window_RR = window_RR.clone(value=window_RR.value() / norm)  # normalize such that ell, ell' = 0, 0 is 1. at s = 0
    if output_fn is not None and jax.process_index() == 0:
        fn = Path(output_fn).parent / 'covariance_windows' / f'RR_{fn.name}'
        window_RR.write(fn)
    
    covs_analytical = compute_spectrum2_covariance(windows, theory, delta=0.4)
    covariance = covs_analytical[0].clone(value=sum(cov.value() for cov in covs_analytical))
    covariance = covariance.at.observable.at(fields=('pre', 'pre')).match(spectrum_pre)

    spectrum_post = covariance.observable.get(fields=('post', 'post'))
    from jaxpower.cov2 import project_to_correlation
    from lsstypes import Mesh2CorrelationPole, Mesh2CorrelationPoles
    
    projector = project_to_correlation(sedges, spectrum_post)
    nmodes = 4 * np.pi / 3. * (sedges[:, 1]**3 - sedges[:, 0]**3)
    norm = spectrum_pre.get(ells=0).values('norm').mean()
    correlation_post = Mesh2CorrelationPoles([Mesh2CorrelationPole(s=s, s_edges=sedges, num_raw=np.zeros_like(s), nmodes=nmodes, norm=norm * np.ones_like(s), ell=ell) for ell in spectrum_post.ells])
    correlation_post.attrs.update({name: value for name, value in kw.items() if name in ['z', 'b1', 'smoothing_radius']})

    from scipy.linalg import block_diag
    rotation = [np.eye(spectrum_pre.size), projector]
    rotation = block_diag(*rotation)
    observable = ObservableTree([spectrum_pre, correlation_post], observables=['spectrum', 'correlationrecon'])
    covariance = covariance.clone(value=rotation.dot(covariance.value()).dot(rotation.T), observable=observable)

    def compute_shotnoise_contribution(observable, QS):

        from jax import numpy as jnp
        from jaxpower.utils import legendre_product

        def get_wj(ww, sedges1, sedges2, q1, q2):
            s = ww.get(0).coords('s')
            s1, s2 = np.mean(sedges1, axis=-1), np.mean(sedges2, axis=1)
            w = sum(legendre_product(q1, q2, q) * ww.get(q).value().real if q in ww.ells else jnp.zeros(()) for q in list(range(abs(q1 - q2), q1 + q2 + 1)))
            def get_volume(*edges):
                volume = 4. / 3. * np.pi * (edges[1]**3 - edges[0]**3)
                return jnp.where(volume < 0., 0., volume)

            sedges_inter = jnp.maximum(sedges1[:, 0], sedges2[None, :, 0]), jnp.minimum(sedges1[:, 1], sedges2[:, 1])
            volume_inter = get_volume(*sedges_inter)
            volume_joint = get_volume(*sedges1.T)[:, None] * get_volume(*sedges2.T)
            return volume_inter / volume_joint * jnp.diag(w)  # FIXME

        pole1 = pole2 = observable
        ills1 = list(range(len(pole1.ells)))
        ills2 = list(range(len(pole2.ells)))

        def init():
            return [[np.zeros((len(pole1.get(pole1.ells[ill1]).coords('s')), len(pole2.get(pole2.ells[ill2]).coords('s')))) for ill2 in ills2] for ill1 in ills1]

        cov_SS = init()
        for ill1, ill2 in itertools.product(ills1, ills2):
            ell1, ell2 = pole1.ells[ill1], pole2.ells[ill2]
            sedges1, sedges2 = pole1.get(ell1).edges('s'), pole2.get(ell2).edges('s')
            cov_SS[ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * get_wj(QS.select(s=sedges1), sedges1, sedges2, ell1, ell2)
        return np.block(cov_SS)

    # Adding back shot noise component
    QS = window.get('SS').get(fields=(0, 0, 0, 0))
    offset = [np.zeros((spectrum_pre.size,) * 2), compute_shotnoise_contribution(correlation_post, QS)]
    offset = block_diag(*offset)
    covariance = covariance.clone(value=covariance.value() + offset)

    rotation = [np.eye(spectrum_pre.size), np.linalg.inv(window_RR.value())]  # deconvolve from the window
    rotation = block_diag(*rotation)
    covariance = covariance.clone(value=rotation.dot(covariance.value()).dot(rotation.T), observable=observable)

    value = covariance.value()
    assert np.allclose(value, value.T)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        covariance.write(output_fn)
    return covariance


def compute_jaxpower_covariance_mesh2_spectrum_pre_bao_post(output_fn, covariance_fn, tracer='LRG'):
    import jax
    from jax import numpy as jnp
    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerCorrelationFunctionMultipoles
    covariance = types.read(covariance_fn)
    covariance = covariance.at.observable.at(observables='correlationrecon').get(ells=[0, 2])
    correlation_post = covariance.observable.get(observables='correlationrecon')
    attrs = correlation_post.attrs
    s = correlation_post.get(ells=0).coords('s')
    template = BAOPowerSpectrumTemplate(z=attrs['z'], fiducial='DESI', apmode='qisoqap')
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(s=s, template=template, smoothing_radius=attrs['smoothing_radius'], broadband='pcs2', ells=list(correlation_post.ells), mode='recsym')
    theory.init.params['sigmas'].update(fixed=True)
    sigmapar, sigmaper = {'BGS': (8, 3)}.get(tracer[:3], (6, 3))
    print(sigmapar, sigmaper)
    theory.init.params['sigmapar'].update(value=sigmapar)
    theory.init.params['sigmaper'].update(value=sigmaper)
    correlation = theory(b1=attrs['b1'])
    params = theory.varied_params
    values = {param.name: param.value for param in params}
    values['b1'] = attrs['b1']
    values = np.array(list(values.values()))
    jac = jax.jacfwd(lambda values: jnp.ravel(theory({param.name: value for param, value in zip(params, values)})))(values).T
    covariance_correlation_post = covariance.at.observable.get(observables='correlationrecon').value()
    inv = np.linalg.inv(covariance_correlation_post)
    projector = np.linalg.inv(jac.dot(inv).dot(jac.T)).dot(jac.dot(inv))

    spectrum_pre = covariance.observable.get(observables='spectrum')
    rotation = [np.eye(spectrum_pre.size), projector]
    from scipy.linalg import block_diag
    rotation = block_diag(*rotation)
    value = rotation.dot(covariance.value()).dot(rotation.T)
    bao_params = ['qiso', 'qap']
    index = np.concatenate([np.arange(spectrum_pre.size), [spectrum_pre.size + params.index(name) for name in bao_params]])
    value = value[np.ix_(index, index)]
    observable = ObservableTree([spectrum_pre, ObservableTree([ObservableLeaf(value=np.ones(1))] * 2, parameters=bao_params)], observables=['spectrum', 'baorecon']) 
    covariance = covariance.clone(value=value, observable=observable)
    value = covariance.value()
    assert np.allclose(value, value.T)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        covariance.write(output_fn)
    return covariance