import time
from datetime import datetime, timedelta

period = timedelta(seconds = 1)
next_time = datetime.now() + period
seconds = 0

while True:
    if next_time < datetime.now():
        seconds += 1
        next_time += period
        print(seconds)
    if seconds == 5:
        break
