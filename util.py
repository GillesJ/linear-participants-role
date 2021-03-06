"""

"""
import os
from cbrole_logging import setup_logging
import logging

setup_logging()


def ensure_dir(dirp):
    if not os.path.exists(dirp):
        os.mkdir(dirp)


def remove_indexes(l, idc):
    return [i for j, i in enumerate(l) if j not in idc]


def flatten(l):
    return [item for sublist in l for item in sublist]


def send_text_message(msg):
    from twilio.rest import Client

    # Find these values at https://twilio.com/user/account
    account_sid = "XXXXX"
    auth_token = "XXXX"
    to_number = "XXXXX"
    from_number = "XXXXX"
    client = Client(account_sid, auth_token)

    message = client.api.account.messages.create(
        to=to_number, from_=from_number, body=msg
    )


def gen_dict_extract(key, var):
    if hasattr(var, "iteritems"):
        for k, v in var.iteritems():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result
