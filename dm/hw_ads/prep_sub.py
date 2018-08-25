f = open ("probabilities.txt")

s = open ("sub.csv", 'w')
s.write ('Id,Click\n')

i = 1
for l in f.readlines ():
	s.write  (f"{i},{l}")
	i += 1
