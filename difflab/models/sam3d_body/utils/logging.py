# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging


def get_pylogger(name=__name__) -> logging.Logger:
    return logging.getLogger(name)
