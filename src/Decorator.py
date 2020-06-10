import time
import logging


logger = logging.getLogger(__name__)


def timeit(method):
    """
    froked from https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)

        else:
            print(f"timed {method.__name__}  {(te - ts):.3f} s")
            # logger.info(f"timed {method.__name__}  {(te - ts) * 1000:.2f} ms")

        return result

    return timed
