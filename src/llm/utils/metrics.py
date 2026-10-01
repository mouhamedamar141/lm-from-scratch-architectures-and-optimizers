"""Mesures simples utilisées dans les expériences."""
import math


def perplexity(loss):
    return math.exp(min(float(loss), 80.0))


def bits_per_char(loss):
    return float(loss) / math.log(2)
