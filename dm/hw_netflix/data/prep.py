f = open ('cheat.txt', 'r')
res = open ('res.txt', 'w')

res.write ("StringId,Mark\n")
i = 1
for l in f.readlines ():
	res.write ("{},{}\n".format (i, l.strip ()))
	i += 1
