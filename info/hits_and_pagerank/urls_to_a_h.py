urls = open ('urls.txt')

h_scores = open ('h_scores.txt', 'w')
a_scores = open ('a_scores.txt', 'w')

for line in urls.readlines():
	doc = int (line.split ('\t')[0])
	h_scores.write ('{}\t1\n'.format (doc))
	a_scores.write ('{}\t1\n'.format (doc))
	#print doc
