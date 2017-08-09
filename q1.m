clear ; %clears all the variables 
close all; %closes extra windows
clc %clears the screen
% ============================================= Part 1: TEXT CLASSIFICATION  ============================================

% -------------------------------------------(a) Training & Test Data Accuracy------------------------------------------

A = importdata('r8-train-all-terms.txt');
B = importdata('r8-test-all-terms.txt');
  
m_train = length(A);		%no. of training examples
m_test = length(B);			%no. of test examples

trainD = [];			% m_train x 2 array to hold training data
testD = [];				% m_test x 2 array to hold test data

for i = 1 : m_train
	trainD = [trainD; strsplit(strtrim(A{i}),'\t')];		%splits the string A{i} based on tab delimiter
end
for i = 1 : m_test
	testD = [testD; strsplit(strtrim(B{i}),'\t')];		%splits the string B{i} based on tab delimiter
end

y = unique(trainD(:,1));		%array to hold the labels
dict = [];			%array to hold the dictionary words

for i = 1 : m_train
	tmp = trainD(i,2);
	tmp = unique((char(strsplit(tmp{1},' '))), 'rows');
	dict = unique(strvcat(dict,tmp), 'rows');
end
for i = 1 : m_test
	tmp = testD(i,2);
	tmp = unique((char(strsplit(tmp{1},' '))), 'rows');
	dict = unique(strvcat(dict,tmp), 'rows');
end

V = size(dict,1);		%no. of words in the dictionary
nj = zeros(V,m_train);		%matrix to store the frequency of each word in every training example

for i = 1 : m_train			%loop to fill the elements of nj matrix 
	tmp = trainD(i,2);
	tmp = sort((strsplit(tmp{1},' '))');
	c = 1;
	for j = 1 : (length(tmp)-1)
		if (strcmp(tmp(j),tmp(j+1)) == 1)
			c = c + 1;
		else
			idx = strcmp(tmp(j), dict);	%vector of size same as dict that stores 1 on those indices for which the string matches tmp(j) and rest 0
			id = find(idx == 1);
			nj(id,i) = nj(id,i) + c;			%total no. of jth words present in the ith example
			c = 1;
		end
	end
	idx = strcmp(tmp(length(tmp)), dict);
	id = find(idx == 1);
	nj(id,i) = nj(id,i) + c;
end

nc = length(y); 		%no. of classes
yi = zeros(nc, m_train);		%matrix that stores 1 in (i,j) if jth example of training data has ith class

for i = 1 : m_train			%loop to fill the elements of yi
	idx = strcmp(trainD(i,1), y);
	yi(find(idx == 1), i) = 1;
end

%%---------------------------Training----------------------------------------------
phi = zeros(nc,V) + 1;		%matrix to hold the probabilities(parameters)
phi_d = zeros(nc,1) + V;			%matrix to hold the denominator of probabilities(parameters)

py = log(sum(yi,2)/m_train);		%nc-sized vector that holds the probability of all the classes (py : probability of y)

for i = 1 : nc				%loop to calculate the log of probabilities(parameters) 
	idx = find(yi(i,:) == 1);
	phi_d(i) = phi_d(i) + sum(sum(nj(:,idx)));
%	phi(i,:) = log(sum(nj(:,idx),2)/phi_d(i));
	for j = 1 : V
		phi(i,j) = phi(i,j) + sum(nj(j,idx));
	end
	phi(i,:) = log(phi(i,:)/phi_d(i));
end

%%---------------------------Prediction------------------------------------------------

%Predicting training data
c = 0;
for i = 1 : m_train			
	tmp = trainD(i,2);
	tmp = (strsplit(tmp{1},' '))';
	tmp2 = zeros(length(tmp),1);	%vector that stores the index(as in dict) of each word of the document
	for j = 1 : length(tmp)			%loop to fill the elements of tmp2
		idx = strcmp(tmp(j),dict);
		tmp2(j) = find(idx == 1);
	end
	y_pred = predict(tmp2,phi,py);
	if (strcmp(trainD(i,1),y(y_pred)) == 1)
		c = c + 1;
	end
end
TrainDataAccuracy = (c * 100)/m_train

%Predicting test data
c = 0;
pred_test = zeros(m_test,1);			%vector to store the predicted label of each example in the test data
for i = 1 : m_test			
	tmp = testD(i,2);
	tmp = (strsplit(tmp{1},' '))';
	tmp2 = zeros(length(tmp),1);	%vector that stores the index(as in dict) of each word of the document
	for j = 1 : length(tmp)			%loop to fill the elements of tmp2
		idx = strcmp(tmp(j),dict);
		tmp2(j) = find(idx == 1);
	end
	y_pred = predict(tmp2,phi,py);
	pred_test(i) = y_pred;
	if (strcmp(testD(i,1),y(y_pred)) == 1)
		c = c + 1;
	end
end
TestDataAccuracy = (c * 100)/m_test
toc;
% -------------------------------------------(b) Random & Majority Prediction--------------------------------------------

cy = zeros(nc,1);
for i = 1 : m_train			%Storing the frequency of each class (cy : count of y)
	idx = strcmp(trainD(i,1), y);
	id = find(idx == 1);
	cy(id) = cy(id) + 1;
end

cr = 0;
cm = 0;

for i = 1 : m_test			
	y_predr = randi(8);					%Random Prediction
	y_predm = find(cy == max(cy));		%Majority Prediction
	if (strcmp(testD(i,1),y(y_predr)) == 1)
		cr = cr + 1;
	end
	if (strcmp(testD(i,1),y(y_predm)) == 1)
		cm = cm + 1;
	end
end
RandPredAccuracy = (cr * 100)/m_test
MajPredAccuracy = (cm * 100)/m_test

% -----------------------------------------------(c) Confusion Matrix---------------------------------------------------

yi_test = zeros(nc, m_test);		%matrix that stores 1 in (i,j) if jth example of test data has ith class

for i = 1 : m_test				%loop to fill the elements of yi_test
	idx = strcmp(testD(i,1), y);
	yi_test(find(idx == 1), i) = 1;
end

confMat = zeros(nc,nc);			%Confusion Matrix
for i = 1 : nc				%loop to fill the elements of Confusion Matrix
	idx = find(yi_test(i,:) == 1);
	for j = 1 : length(idx)
		confMat(i,pred_test(idx(j))) = confMat(i,pred_test(idx(j))) + 1;
	end
end