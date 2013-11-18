'''
Let's explain the logistic regression by example. Consider you are the administrator of a university department and you want to determine each applicant's chance of admission based on their results on two exams. You have the historical data from previous applicants that you can use as a trainning set for logistic regression.  For each training example, you have the applicant's scores on two exams and the admissions decision.   We will use logistic regression to build this model that estimates the probability of admission based the scores from those two exams.
'''

from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
 
#load the dataset
data = loadtxt('ex2data1.txt', delimiter=',')
 
X = data[:, 0:2] # scores
y = data[:, 2]  #final value Is Admitted?
 
pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()


