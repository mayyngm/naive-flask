spellingd = {}
with open('spelling.csv') as f:
   for line in f:
       parts = line.split(',')

       parts[0] = parts[0].strip()
       parts[1] = parts[1].strip()
       
       spellingd[parts[0]] = parts[1]
       if not line:
          break

test = 'bentar saya lagi ketauan'
parts = test.split(' ')

hasil = ''
for p in parts:
    if p in spellingd:
        hasil = hasil+spellingd[p]+' '
    else:
        hasil = hasil+p+' '
print(hasil)
