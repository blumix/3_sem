import sys

f = open ("preds.txt", "r")
res = open ("pred.csv","w")

res.write ("Id,Click\n")

i = 1
for line in f.readlines ():
    res.write ("{},{}".format (i, line))
    i += 1
