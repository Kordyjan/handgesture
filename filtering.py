from os import remove, listdir
from random import shuffle

ROOT = "data"

for category in listdir(f"{ROOT}/learning"):
    files = listdir(f"{ROOT}/learning/{category}")
    shuffle(files)
    for file in files[:600]:
        remove(f"{ROOT}/learning/{category}/{file}")
    for file in files[600:]:
        remove(f"{ROOT}/validation/{category}/{file}")
