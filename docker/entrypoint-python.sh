#!/bin/bash
# nvidia-cusolver-cu12 is outside TensorFlow's own library search path, so list every nvidia/*/lib
export LD_LIBRARY_PATH=$(python3 -c "import nvidia, os, glob; print(':'.join(glob.glob(os.path.join(os.path.dirname(nvidia.__file__), '*', 'lib'))))"):${LD_LIBRARY_PATH}

exec "$@"
