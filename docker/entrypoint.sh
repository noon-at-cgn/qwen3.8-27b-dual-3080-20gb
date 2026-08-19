#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   serve     serve.sh (the only mode this repo ships — see README)
#   prepare   docker/prepare.sh (download + requantize the model into /app/models)
#   verify    verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, verify.sh --no-server runs and aborts on FAIL (model not
# requantized, patches missing, ...); VERIFY=0 skips that.
set -e
cd /app
cmd=${1:-serve}; shift || true
case "$cmd" in
  serve)
    if [ "${VERIFY:-1}" != "0" ]; then
      bash verify.sh --no-server || { echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"; exit 1; }
    fi
    exec bash serve.sh "$@" ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  verify)  exec bash verify.sh "$@" ;;
  *)       exec "$cmd" "$@" ;;
esac
