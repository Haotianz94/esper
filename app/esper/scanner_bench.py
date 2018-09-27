from scannertools import shot_detection, kube
from esper.prelude import ScannerWrapper, ScannerSQLTable, Timer, Notifier, now, pcache, log
from query.models import Video
from esper.kube import cloud_config
from threading import Thread, Condition
from attr import attrs, attrib
from pprint import pprint
from storehouse import StorehouseException
import random
import traceback
import subprocess as sp
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from time import strftime

notifier = Notifier()

@attrs(frozen=True)
class ScannerJobConfig:
    io_packet_size = attrib(type=int)
    work_packet_size = attrib(type=int)
    batch = attrib(type=int, default=1)

class TestFailure(Exception):
    pass


def bench(name, videos, run_pipeline, configs, sample_size=15, force=False, no_delete=False):

    def run_name(cluster_config, job_config):
        worker_type = cluster_config.worker.type
        return '{name}-{cpu}cpu-{mem}mem-{batch}batch-{wpkt}wpkt-{iopkt}iopkt-{ldwk}ldwk-{svwk}svwk-{vid}vid'.format(
            name=name,
            cpu=worker_type.get_cpu(),
            mem=worker_type.get_mem(),
            batch=job_config.batch,
            wpkt=job_config.work_packet_size,
            iopkt=job_config.io_packet_size,
            ldwk=cluster_config.num_load_workers,
            svwk=cluster_config.num_save_workers,
            vid=sample_size)

    def run_config(videos, cluster, job_config):
        db_wrapper = ScannerWrapper.create(cluster=cluster, enable_watchdog=False)
        db = db_wrapper.db

        # Start the Scanner job
        log.info('Starting Scanner job')
        run_pipeline(db, videos, detach=True, batch=job_config.batch, run_opts={
            'io_packet_size': job_config.io_packet_size,
            'work_packet_size': job_config.work_packet_size,
        })

        # Wait until it succeeds or crashes
        start = now()
        log.info('Monitoring cluster')
        result, metrics = cluster.monitor(db)
        end = now() - start

        # If we crashed:
        if not result:

            # Restart the cluster if it's in a bad state
            cluster.start()

            raise TestFailure("Out of memory")

        # Write out profile if run succeeded
        outputs = run_pipeline(db, videos, no_execute=True)
        try:
            outputs[0]._column._table.profiler().write_trace(
                '/app/data/traces/{}.trace'.format(run_name(cluster.config(), job_config)))
        except Exception:
            log.error('Failed to write trace')
            traceback.print_exc()

        return end, pd.DataFrame(metrics)

    def test_config(videos, cluster, job_config):
        time, metrics = run_config(videos, cluster, job_config)

        if time is not None:
            price_per_hour = cluster.config().price()
            price_per_video = (time / 3600.0) * price_per_hour / float(sample_size)
            return price_per_video, metrics
        else:
            return None

    sample = range(sample_size)
    N = len(videos)
    results = []

    for (cluster_config, job_configs) in configs:

        # Only bring up the cluster if there exists a job config that hasn't been computed
        if not force and all([pcache.has(run_name(cluster_config, job_config)) for job_config in job_configs]):
            results.append([pcache.get(run_name(cluster_config, job_config)) for job_config in job_configs])

        else:
            with kube.Cluster(cloud_config, cluster_config, no_delete=no_delete) as cluster:
                log.info('Cluster config: {}'.format(cluster_config))

                def try_config(job_config):
                    log.info('Job config: {}'.format(job_config))
                    try:
                        return test_config([videos[i] for i in sample], cluster, job_config)
                    except TestFailure as e:
                        return (str(e), None)
                    except Exception as e:
                        return (traceback.format_exc(), None)

                def try_config_cached(job_config):
                    return pcache.get(run_name(cluster_config, job_config), force=force, fn=lambda: try_config(job_config))

                results.append(list(map(try_config_cached, job_configs)))

    def plot(metrics, name):
        ax = metrics.plot('TIME', name)
        ax.set_title(name)
        ax.set_ylabel('Percent')
        ax.set_xlabel('Sample')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig('/tmp/graph.svg')
        fig.clf()
        return open('/tmp/graph.svg', 'r').read()

    report_template = '''
    <!DOCTYPE html>
    <html>
      <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css" integrity="sha384-PsH8R72JQ3SOdhVi3uxftmaW6Vc51MKb0q5P2rRUpPvrszuE4W1povHYgTpBfshb" crossorigin="anonymous">
        <style>
          svg {{ width: 50%; margin: 0; float: left; }}
          p {{ margin-bottom: 0; }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Scanner benchmark report</h1>
          {report}
        </div>
      </body>
    </html>
    '''

    blocks = ''
    for ((cluster_config, job_configs), cluster_results) in zip(configs, results):
        for (job_config, (job_result, metrics)) in zip(job_configs, cluster_results):
            if metrics is None:
                blocks += '<div><h3>{name}</h3><p>{result}</p></div>'.format(
                    name=run_name(cluster_config, job_config), result=job_result)
                continue

            cpu = plot(metrics, 'CPU%')
            mem = plot(metrics, 'MEMORY%')
            block = '''
            <div>
              <h3>{name}</h3>
              <p>${result:.04f}/video, estimated ${total:d} total</p>
              <div>
                {cpu}
                {mem}
              </div>
            </div>
            '''.format(
                name=run_name(cluster_config, job_config),
                result=job_result,
                total=int(job_result * N),
                cpu=cpu,
                mem=mem)
            blocks += block

    report = report_template.format(report=blocks)

    with open('/app/data/benchmarks/benchmark-{}.html'.format(strftime('%Y-%m-%d-%H-%M')), 'w') as f:
        f.write(report)

    # Collect all traces into a tarfile
    sp.check_call('cd /app/data && tar -czf bench.tar.gz traces benchmarks', shell=True)

    # Let desktop know bench is complete, and should download benchmark files
    notifier.notify('Benchmark complete', action='bench')
