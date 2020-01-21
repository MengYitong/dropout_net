import numpy as np
import os
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

print("now =", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y%m%d_%H%M%S")
print("date and time =", dt_string)
