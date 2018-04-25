#!/usr/bin/env python

"""storing and accessing history of stats metrics"""

import logging
from utils import as_py_scalar

LOG = []

def clear_log():
    global LOG
    LOG = []


def get_the_log():
    global LOG
    return LOG


def set_the_log(l):
    global LOG
    LOG = l


def insert_log(it, k, v, log=None):
    global LOG
    if log is None:
        if LOG is None:
            LOG = []
        log = LOG

    v = as_py_scalar(v)

    if len(log) > 0:
        last = log[-1]
        if last['it'] > it:
            msg = 'trying to change history at %d, current is %d' % (it, last['it'])
            logging.error(msg)
            # raise ValueError('trying to change history at %d, current is %d' % (it, last['it']))
            return
        if last['it'] != it:
            last = {'it': it}
            log.append(last)
    else:
        last = {'it': it}
        log.append(last)
    last[k] = v


def get_latest_log(what, default=None, log=None):
    global LOG
    if log is None:
        log = LOG

    latest_row = []
    latest_it = -1
    for row in log[::-1]:
        if row['it'] > latest_it and what in row:
            latest_it = row['it']
            latest_row = row

    if latest_it == -1 or what not in latest_row:
        if default is not None:
            return default, 0
        msg = 'no such key in log: %s' % what
        logging.error(msg)
        raise ValueError('no such key in log: %s' % what)

    return latest_row[what], latest_it


def get_log(what, log=None):
    global LOG
    if log is None:
        log = LOG
    vals = [x[what] for x in log if what in x]
    its = [x['it'] for x in log if what in x]
    return vals, its


def list_log_keys(log=None):
    global LOG
    if log is None:
        log = LOG
    keys = set()
    for entry in log:
        keys |= set(entry.keys())
    return sorted(keys)
