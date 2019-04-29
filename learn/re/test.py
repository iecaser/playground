import re
a = 'abab'
p = '[a|ab]'
c = re.compile(p)
r = c.findall(a)
print(r)
