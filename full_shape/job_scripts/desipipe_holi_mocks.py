import numpy as np

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

setup_logging()

queue = Queue('holi_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/holi_mocks/slurm-%j.out', 'slurm_outputs/holi_mocks/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi') #, command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:10:00',
                            mpiprocs_per_worker=4, output=output, error=error, constraint='gpu'))


@tm.python_app
def run_stats(tracer='LRG', imocks=[451], stats=['mesh2_spectrum']):
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    sys.path.insert(0, '../')
    import tools
    from compute_fiducial_stats import compute_fiducial_stats_from_options, fill_fiducial_options
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    jax.distributed.initialize()
    setup_logging()
    meas_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    cache = {}
    for zrange in tools.propose_fiducial('zranges', tracer):
        for imock in imocks:
            for region in ['NGC', 'SGC']:
                options = dict(catalog=dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=imock), mesh2_spectrum={'cut': True, 'auw': True})
                options = fill_fiducial_options(**options)
                compute_fiducial_stats_from_options(stats, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), **options)
    jax.distributed.shutdown()


if __name__ == '__main__':

    imocks = 451 + np.arange(5)
    for tracer in ['LRG', 'ELG_LOP', 'QSO'][:1]:
        run_stats(imocks=imocks)
