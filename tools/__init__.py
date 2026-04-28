"""cppmega developer tools namespace.

Keep this package extendable because Megatron-LM also ships a top-level
``tools`` package.  Local cppmega paths come first in launchers/tests, but
extending the package path lets imports still resolve Megatron tools when a
script needs them.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
