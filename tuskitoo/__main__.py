"""Run `python -m tudspek`.

Allow running Tudspek, also by invoking
the python module:

`python -m tudspek`

This is an alternative to directly invoking the cli that uses python as the
"entrypoint".
"""

from __future__ import absolute_import

from tudspek.cli import main

if __name__ == "__main__":  # pragma: no cover
    main(prog_name="tudspek")  # pylint: disable=unexpected-keyword-arg
