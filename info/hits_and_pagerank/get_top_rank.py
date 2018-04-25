f_r = open ("rank.txt")
f_urls = open ("urls.txt")
urls = {}
for u in f_urls.readlines ():
    u = u.strip ().split ('\t')
    urls[u[0]] = u[1]

r = {}
for l in f_r.readlines ():
    l = l.strip ().split ('\t')
    r[l[1]] = float (l[0].lower ())

sorted_r = sorted(r.items(), key=lambda x:x[1])

print ("Top Page Rank:")
for s in sorted_r[-30:]:
    print (urls[s[0]])

