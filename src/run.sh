export PYTHONPATH=`pwd`/external/gym-pcgrl:`pwd`:`pwd`/../Evocraft-py

if [ -f $1 ]; then
    python -u  "$@"
else
    python -u  ../"$@"
fi

