from typing import Callable

from sqlalchemy.exc import OperationalError


def retry_on_operational_failure(fun: Callable, retry_num: int = 1) -> Callable:
    """Decorator for catching sqlalchemy.exc.OperationalError,
    which is typically emitted when a connection has been disconnected by the host.
    """

    assert retry_num > 0, 'Need at least one retry'

    def _wrapper(self, *args, **kwargs):
        i = 0
        while i <= retry_num:
            try:
                return fun(self, *args, **kwargs)
            except OperationalError as e:
                i += 1

                if i > retry_num:
                    raise e

                # print('Restart session for retry')
                self.restart_session()

    return _wrapper
