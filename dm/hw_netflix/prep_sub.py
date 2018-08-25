def form_submission(test_pred, name):
	d = {'StringID': range(1, len(test_pred) + 1), 'Mark':  test_pred}
	res = pd.DataFrame(data=d)
	cols = ['StringID', 'Mark']
	res = res[cols]
	res.to_csv(name+'.csv', index=False)

with open('out.txt') as f:
	lines = f.readlines()

for i, line in enumerate(lines):
	res.append(float(line.strip()))

form_submission(res, "submission_1")
