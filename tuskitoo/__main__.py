"""Run `python -m tuskitoo`.

Allow running Tudspek, also by invoking
the python module:

`python -m tuskitoo`

This is an alternative to directly invoking the cli that uses python as the
"entrypoint".
"""

from __future__ import absolute_import

from tuskitoo.cli import main

if __name__ == "__main__":  # pragma: no cover
    main(prog_name="tuskitoo ")  # pylint: disable=unexpected-keyword-arg
