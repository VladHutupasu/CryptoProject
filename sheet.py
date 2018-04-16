#!/usr/bin/python

s="""\
subject--- marks

english---42

maths---48

science---49"""

table=['<html><body><table border="1">']
for line in s.splitlines():
    if not line.strip():
        continue
    table.append(r'<tr><td>{}</td><td>{}</td></tr>'.format(*line.split('---')))

table.append('</table></body></html>')
print(''.join(table))