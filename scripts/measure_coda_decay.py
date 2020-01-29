import argparse
import h5seis
import multiprocessing as mp
import numpy as np
import obspy
import os
import pandas as pd
import scipy
import scipy.signal
import scipy.optimize
import seispy
import sys
import warnings


warnings.filterwarnings('ignore', module='obspy.signal.filter')
warnings.filterwarnings('ignore', module='numpy')
warnings.filterwarnings('ignore', module='scipy')
logger = seispy.logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dbin', type=str, help='input database')
    parser.add_argument('rays', type=str, help='input ray parameters')
    parser.add_argument('wfs', type=str, help='input waveforms')
    parser.add_argument('outfile', type=str, help='output file')
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    parser.add_argument(
        '-l', '--logfile',
        type=str,
        default=f'{script_name}.log',
        help='log file'
    )
    parser.add_argument(
        '-n', '--nthreads',
        type=int,
        default=1,
        help='number of threads'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='verbose'
    )
    parser.add_argument(
        '-a', '--evid_start',
        type=int,
        default=-1,
        help='event index start'
    )
    parser.add_argument(
        '-b', '--evid_end',
        type=int,
        default=-1,
        help='event index end'
    )
    return (parser.parse_args())


def main():
    _args = dict()
    _args['clargs'] = parse_args()
    seispy.logging.configure_logger(
        __name__,
        _args['clargs'].logfile,
        verbose=_args['clargs'].verbose
    )
    logger.info('Begin')
    _args['df'] = read_db(_args)
    init_outfile(_args['clargs'].outfile)
    event_pool(_args)
    logger.info('Complete')


def chunkify(iterable, size):
    idx_start = 0
    while idx_start < len(iterable):
        idx_end = idx_start + size
        yield (iterable[idx_start: idx_end])
        idx_start = idx_end


def event_pool(_args):
    nthreads = _args['clargs'].nthreads
    logger.info(f'Intializing Worker Pool with {nthreads} threads')

    df = _args['df']
    idxs = df.index.unique().values

    with mp.Pool(nthreads) as pool:
        for chunk in chunkify(idxs, nthreads):
            results = pool.map(
                process_event,
                [
                    (idx, df.loc[idx], _args['clargs'].wfs, _args['clargs'].rays)
                    for idx in chunk
                ]
            )
            append_results(results, _args)


def init_outfile(outfile):
    logger.info('Initializing outfile')
    with pd.HDFStore(outfile) as store:
        store['coda'] = pd.DataFrame()
    logger.info('Outfile initialized')

def append_results(results, _args):
    logger.info('Appending results')
    with pd.HDFStore(_args['clargs'].outfile) as store:
        store['coda'] = store['coda'].append(
            pd.concat(results, ignore_index=True),
            ignore_index=True
        )
    logger.info('Results appended')

def process_event(args):
    (event_id, origin_id), df, wffile, rayfile = args
    origin_time = df.iloc[0]['time']
    df = df.sort_values('sta').set_index('sta')
    logger.info(f'Processing event #{event_id}')
    try:
        st = get_wfs(event_id, wffile)
        rays_p, rays_s = get_rays(origin_id, rayfile)
    except Exception as exc:
        logger.debug(exc)
        return (None)
    df_out = pd.DataFrame()
    for station in df.index.unique().values:
        if station not in rays_p.index or station not in rays_s.index:
            logger.debug(f'Station {station} not found in rays file for event #{event_id}')
            continue
        try:
            dt = measure_coda_decay(
                st.select(station=station).copy(),
                origin_time,
                rays_p.loc[station, 'tt'],
                rays_s.loc[station, 'tt']
            )
            if dt is not None:
                df_out = df_out.append(
                    pd.DataFrame(
                        {
                            'orid': [origin_id],
                            'evid': [event_id],
                            'sta':  [station],
                            'dt':   [dt],
                        }
                    ),
                    ignore_index=True
                )
        except Exception as exc:
            logger.warning(exc)
            continue
    logger.debug(f'Finished processing event #{event_id}')
    return (df_out)


def measure_coda_decay(st, origin_time, tt_p, tt_s):
    npts = 5
    try:
        tr0 = st.select(channel='??Z')[0]
    except IndexError:
        return (None)

    origin_time = obspy.UTCDateTime(origin_time)

    atime_p = origin_time + tt_p
    atime_s = origin_time + tt_s

    tr0.filter('bandpass', freqmin=1, freqmax=20)

    tr_p = tr0.slice(
        starttime=atime_p-0.5,
        endtime=atime_p+2.
    )

    times      = tr_p.times()
    analytic   = np.abs(scipy.signal.hilbert(tr_p))
    peaks, _   = scipy.signal.find_peaks(analytic)
    argmax     = np.argmax(analytic[peaks])
    xx         = times[peaks[argmax:argmax+npts]]
    yy         = analytic[peaks[argmax:argmax+npts]]
    xmax       = xx.max()
    dx         = xx.max() - xx.min()
    if dx == 0:
        return (None)
    zz         = (xx - xx.min()) / dx
    popt, pcov = scipy.optimize.curve_fit(lambda x, a, b: a * np.exp(b * x), zz, yy)
    dt_p         = np.log(0.5) / popt[1] * dx

    return (dt_p)

def get_rays(origin_id, rayfile):
    ray_tag = orid_to_tag(origin_id)
    with pd.HDFStore(rayfile, mode='r') as store:
        rays_p = store[ray_tag + '/P']
        rays_s = store[ray_tag + '/S']
        rays_p = rays_p[~rays_p.index.duplicated()]
        rays_s = rays_s[~rays_s.index.duplicated()]
    return (rays_p, rays_s)

def get_wfs(event_id, wffile):
    wf_tag = f'event_{int(event_id)}'
    with h5seis.H5Seis(wffile, mode='r') as h5:
        st = h5.get_waveforms_for_tag(wf_tag)
    return (st)


def read_db(_args):
    logger.info('Reading event database')
    db = seispy.pandas.catalog.Catalog(
        _args['clargs'].dbin,
        fmt='hdf5',
        tables=['origin', 'arrival', 'assoc']
    )
    df = db['origin'].merge(
        db['assoc'][['arid', 'orid']],
        on='orid'
    ).merge(
        db['arrival'][['arid', 'sta', 'iphase', 'time']],
        on='arid',
        suffixes=('', '_arrival')
    )
    if _args['clargs'].evid_start is not -1:
        df = df[df['evid'] >= _args['clargs'].evid_start]
    if _args['clargs'].evid_end is not -1:
        df = df[df['evid'] <= _args['clargs'].evid_end]
    df = df.sort_values(
        ['evid', 'orid']
    ).set_index(
        ['evid', 'orid']
    )
    logger.info('Event database read')
    return (df)


def orid_to_tag(orid):
    mod = orid % 10000
    return (f'_{orid-mod}/_{mod}')


if __name__ == '__main__':
    main()
