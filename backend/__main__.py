"""CLI entrypoint for ``python -m backendHelen``."""

from .server import main

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
