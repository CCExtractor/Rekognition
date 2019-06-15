from celery import shared_task,current_task
from numpy import random
from scipy.fftpack import fft

@shared_task
def add(x, y):
    print('*'*(x+y))
    return x + y

