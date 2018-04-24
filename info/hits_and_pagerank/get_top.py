f_a = open ("a.txt")
f_b = open ("b.txt")
f_urls = open ("urls.txt")
urls = {}
for u in f_urls.readlines ():
    u = u.strip ().split ('\t')
    urls[u[0]] = u[1]

a = {}
for l in f_a.readlines ():
    l = l.strip ().split ('\t')
    a[l[0]] = float (l[1].lower ())

b = {}
for l in f_b.readlines ():
    l = l.strip ().split ('\t')
    b[l[0]] = float (l[1].lower ())

sorted_a = sorted(a.items(), key=lambda x:x[1])
sorted_b = sorted(b.items(), key=lambda x:x[1])

print ("Top H:")
for s in sorted_a[-30:]:
    print (urls[s[0]])

print ("Top A:")
for s in sorted_b[-30:]:
    print (urls[s[0]])