import itertools

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
import jax
from jax import numpy as jnp

from . import tools


@jax.jit
def _compute_healpix_map_jax(ra, dec, weights=1., nside: int=256, nest: bool=False):
    import jax_healpy as hp

    hpmap = jnp.zeros(hp.nside2npix(nside))
    pix = hp.vec2pix(nside, ra, dec, lonlat=True, nest=nest)
    return hpmap.at[pix].add(weights)


def _compute_healpix_map_numpy(ra, dec, weights=1., nside: int=256, nest: bool=False):
    hpmap = np.zeros(hp.nside2npix(nside))
    pix = hp.vec2pix(nside, ra, dec, lonlat=True, nest=nest)
    np.add.at(hpmap, pix, weights)
    return hpmap


def compute_angular_density(catalog, nside: int=256, nest: bool=False, backend: str='numpy'):
    """
    Compute an angular (HEALPix) density map from a catalog.

    The catalog may already contain 'RA' and 'DEC' fields; otherwise RA/DEC are
    derived from the 'POSITION' field via tools.cartesian_to_sky. When using the
    'jax' backend the computation happens on JAX arrays with optional sharding.

    Parameters
    ----------
    catalog : Catalog
        Catalog.
    nside : int, optional
        HEALPix nside resolution (default 256).
    nest : bool, optional
        Use NESTED pixel ordering if True (default False).
    backend : {'numpy', 'jax'}, optional
        Backend to use for HEALPix computation. 'numpy' uses NumPy+MPI, 'jax' uses JAX.

    Returns
    -------
    array-like
        HEALPix map containing the weighted counts per pixel.
    """
    if 'RA' not in catalog:
        catalog['RA'], catalog['DEC'] = tools.cartesian_to_sky(catalog['POSITION'])
    ra, dec, weights = catalog['RA'], catalog['DEC'], catalog.get('INDWEIGHT', catalog.ones())
    if backend == 'jax':
        from jaxpower.mesh import create_sharding_mesh, make_array_from_process_local_data
        with create_sharding_mesh() as sharding_mesh:
            ra, dec = [make_array_from_process_local_data(rd, pad='uniform', sharding_mesh=sharding_mesh) for rd in [ra, dec]]
            weights = make_array_from_process_local_data(weights, pad=0., sharding_mesh=sharding_mesh)
            hpmap = _compute_healpix_map_jax(ra, dec, weights=weights, nside=nside, nest=nest)
    else:
        # backend = 'numpy'
        hpmap = _compute_healpix_map_numpy(ra, dec, weights=weights, nside=nside, nest=nest)
        mpicomm = catalog.mpicomm
        hpmap = mpicomm.allreduce(hpmap)
    return hpmap


def compute_redshift_density(catalog, edges=None, backend: str='numpy'):
    """
    Compute a weighted redshift histogram for a catalog.

    Parameters
    ----------
    catalog : Catalog
        Catalog.
    edges : sequence or int or None, optional
        Bin edges for the histogram. If an integer is provided, that number of
        equal-width bins between min(z) and max(z) is constructed.
    backend : {'numpy', 'jax'}, optional
        Backend to use for histogramming. 'numpy' uses NumPy+MPI, 'jax' uses JAX.

    Returns
    -------
    hist, edges
    """
    z, weights = catalog['Z'], catalog.get('INDWEIGHT', catalog.ones())
    if isinstance(edges, int):
        import mpytools as mpy
        cmin, cmax = mpy.cmin(z), mpy.cmax(z)
        edges = np.linspace(cmin, cmax, edges + 1)
    if backend == 'jax':
        from jaxpower.mesh import create_sharding_mesh, make_array_from_process_local_data
        with create_sharding_mesh() as sharding_mesh:
            z = make_array_from_process_local_data(z, pad='uniform', sharding_mesh=sharding_mesh)
            weights = make_array_from_process_local_data(weights, pad=0., sharding_mesh=sharding_mesh)
            hist = jnp.histogram(z, bins=edges, weights=weights)
    else:
        hist = np.histogram(z, bins=edges, weights=weights)[0]
        mpicomm = catalog.mpicomm
        hist = mpicomm.allreduce(hist)
    return hist, edges


def plot_density_projections(get_catalog_fn=tools.get_catalog_fn, read_catalog=tools.read_clustering_catalog,
                             catalog=dict(), divide_randoms: bool | str=False, backend: str='numpy', zedges=None,
                             nside=256, fn=None, **kwargs):
    """
    Plot angular density projections (HEALPix) and optional redshift distributions.

    This convenience function reads one or more catalogs determined by combinations
    of keyword grid `kwargs` and plots the averaged HEALPix maps and, if requested,
    the redshift histogram. Optionally divides data by randoms to produce overdensity.

    Parameters
    ----------
    get_catalog_fn : callable, optional
        Function returning a path or handle given kind='data'/'randoms' and keyword args.
    read_catalog : callable, optional
        Function that reads a catalog given a path.
    catalog : dict, optional
        Base keyword args passed to get_catalog_fn / read_catalog.
    divide_randoms : bool or 'same', optional
        If True divide data maps by randoms maps.
        Pass 'same' to reuse the same randoms.
    backend : {'numpy', 'jax'}, optional
        Backend to use for map/histogram computation.
    zedges : sequence or None, optional
        If provided, compute and plot redshift histograms using these edges.
    nside : int, optional
        HEALPix nside resolution for angular maps.
    fn : str or None, optional
        If provided, save the figure to this filename.
    **kwargs :
        Keyword grid to iterate: keys map to parameter names of :func:`get_catalog_fn` and values to sequences.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the HEALPix projection (and z-histogram if requested).
    """
    nest = False
    with_zhist = zedges is not None
    randoms_hpmap = randoms_zhist = None
    hpmap, zhist = 0., 0.
    ndata = 0
    names, values = zip(*kwargs.items())
    for values in itertools.product(*values):
        fn_kwargs = catalog | dict(zip(names, values))
        data = read_catalog(get_catalog_fn(kind='data', **fn_kwargs), kind='data', **fn_kwargs)
        data_hpmap = compute_angular_density(data, nside=nside, nest=nest, backend=backend)
        if with_zhist:
            data_zhist = compute_redshift_density(catalog, edges=zedges, backend=backend)
        if divide_randoms:
            if not (divide_randoms == 'same' and randoms_hpmap is not None):
                randoms = read_catalog(get_catalog_fn(kind='randoms', **fn_kwargs), kind='randoms', **fn_kwargs)
                randoms_hpmap = compute_angular_density(randoms, nside=nside, nest=nest, backend=backend)
                if with_zhist: randoms_zhist = compute_redshift_density(randoms, edges=zedges, backend=backend)
                del randoms
            data_hpmap = data_hpmap / randoms_hpmap * randoms_hpmap.sum() / data_hpmap.sum()
            data_zhist = data_zhist / randoms_zhist * randoms_zhist.sum() / data_zhist.sum()
        hpmap += data_hpmap
        if with_zhist:
            zhist += data_zhist
        ndata += 1

    hpmap, zhist = hpmap / ndata, zhist / ndata

    fig, lax = plt.subplots(1, 1 + with_zhist, figsize=(10, 4))
    plt.sca(lax[0])
    hp.mollview(hpmap, hold=True, cbar=True, nest=nest)
    if with_zhist:
        ax = lax[0]
        ax.stairs(zhist, zedges)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$n(z)$')
    if fn is not None:
        plt.tight_layout()
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close(plt.gcf())
    return fig