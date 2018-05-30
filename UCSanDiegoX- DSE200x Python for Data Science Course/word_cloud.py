import collections

file=open('98-0.txt',"r",encoding='utf8')

wordcount={}

stop=open("stopwords","r",encoding='utf8')
stopwords=stop.read().split()


for word in file.read().lower().split():
    word = word.replace("."," ")
    word = word.replace(","," ")
    word = word.replace("'"," ")
    word = word.replace("-", " ")
    word = word.replace("(", " ")
    word = word.replace(")", " ")
    word = word.replace('"', " ")
    word = word.replace("“", "")
    word = word.replace("”"," ")
    word = word.split()
    for i in word:
        if i not in stopwords:
            if i not in wordcount:
                wordcount[i] = 1
            else:
                wordcount[i] += 1

d = collections.Counter(wordcount)

for word, count in d.most_common(20):
    print(word, ": ", count)
