from naivebayes import NaiveBayes

nb = NaiveBayes(3, 3)



dataset = [
([0, 0, 1], 1), 
([0, 1, 0], 0),
([0, 1, 1], 1), 
([1, 0, 0], 0),
([1, 1, 0], 0),
([1, 1, 1], 2), 
([1, 0, 1], 2), 
([0, 1, 1], 1),
([0, 1, 1], 1),
([0, 1, 1], 1),
([0, 1, 1], 1),
([0, 1, 1], 2),
([0, 0, 1], 1),
([1, 0, 1], 2),
([1, 1, 0], 0) ]


for i,t in dataset:
	nb.update(i,t)

print nb.class_count
print nb.feature_count

for i,t in dataset:
	print nb.classify(i), t