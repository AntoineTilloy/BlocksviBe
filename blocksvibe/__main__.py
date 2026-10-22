import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):     # see bin/blocks_3d
    os.environ.setdefault(_v, "1")

from .cli import main

sys.exit(main())
