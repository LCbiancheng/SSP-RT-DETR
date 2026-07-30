import contextlib
import os


__all__ = ['is_concise_logging', 'suppress_output']


def is_concise_logging() -> bool:
    return os.getenv('ABLATION_CONCISE', '0') == '1'


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
