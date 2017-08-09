function y_pred = predict(doc,phi,py)
	y_predv = -Inf;
	nc = length(py);
	for i = 1 : nc
		y = sum(phi(i,doc)) + py(i);
		if (y > y_predv)
			y_predv = y;
			y_pred = i;
		end
	end
end