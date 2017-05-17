clear; clc;
% the number of points per axis
N = 50;
% the range where we will compute the function is [-1, 1] both for x and y
x = linspace(-1, 1, N);
y = linspace(-1, 1, N);
[X, Y] = meshgrid(x, y);
f = @(X, Y)(sin(pi*X) + cos(pi*Y));

% plot the function in 3D space
subplot(1, 2, 1);
mesh(X, Y, f(X, Y));
axis([-1, 1, -1, 1, -2.1, 2.1]);
title('Exact Function');
xlabel('x');
ylabel('y');
legend('sin(\pi x) + cos(\pi y)','Location','NorthWest')

% create all possible pairs of the coordinates (2  x  N^2)
coord = combvec(x, y);

% number of samples to use for the training of the NN
M = 200;
% number of samples to use for the testing of the NN
T_M = floor(M * 3 / 7);

assert(M + T_M <= N * N, 'The number of samples used for training and testing must be less or equal to the total number of examples.');

% sample randomly M + T_M pairs of coordinates. Use the M first to train the NN
% and the T_M rest to test its performance (70% train / 30% test)
sample = datasample(coord, M+T_M, 2, 'Replace', false);
X_train = sample(1, 1:M);
Y_train = sample(2, 1:M);
train_output = f(X_train, Y_train);

X_test  = sample(1, M+1:M+T_M);
Y_test  = sample(2, M+1:M+T_M);
test_output = f(X_test, Y_test);

% create a feed forward NN with one hidden layer of 10 neurons
net = feedforwardnet(10);
% don't show the window of the nntraintool
net.trainParam.showWindow = false;
% train the NN
net = train(net, [X_train; Y_train], train_output);
% get the outputs from the NN for the test input
z = net([X_test; Y_test]);
% find the performance of the NN using the test set
perf = perform(net, test_output, z);
fprintf('Mean Squared Error: %f\n', perf);

% plot the estimates of the NN in 3D space, and the points that were used to
% train the NN
subplot(1, 2, 2);
mesh(X, Y, reshape(net([reshape(X, 1, N*N); reshape(Y, 1, N*N)]), N, N));
hold on;
scatter3(X_train, Y_train, train_output, 'r', 'filled');
hold off;
axis([-1, 1, -1, 1, -2.1, 2.1]);
title('Function Estimate by NN');
xlabel('x');
ylabel('y');
legend('NN estimate', 'training samples', 'Location','NorthWest');

% save output plot
h = gcf;    % get current figure handle
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h, 'images/fun_approx', '-dpdf','-r0');
