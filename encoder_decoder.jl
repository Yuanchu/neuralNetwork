
using neuralNet
using MNIST
# using MATLAB
testdata, testlabel = testdata()

# standardize the data by dividing by the maximum
# this is very important. Otherwise the Hessian and gradient calculation will blow up
testdata = testdata / maximum(testdata);

# try just 10 datasets to begin with 
testdata1 = testdata[:, 1:10]

# three layers with 784, 300, and 784 layers respectively
archi = [784 300 784];

# nn contains W's for each sample in the dataset
nn = network(archi);

# nnw contains W's only for one sample in the dataset
nnw = network(archi);

# initiailize weights for nn
initialize_w!(nn, testdata1)

# number of iterations
iter = 1

# training the NN:
# loop for iter iterations
for l = 1:iter
    
    # first do W-update, in a parallel way
    # loop through the dataset
    w = @parallel hcat for k = 1:size(testdata1, 2)
        
        nnw = deepcopy(nn)
    
        # loop through the layers, but only take the W for the k-th sample
        for i = 1:(nn.n - 1)
            nnw.w[i] = deepcopy(nn.w[i][:, k])''
        end
        w_update!(nnw, testdata1[:, k], testdata1[:, k], 1)
        nnw.w
    end
    
    # put W's back into nn, layer by layer
    for layer in 1:(nn.n - 1)
        nn.w[layer] = reduce(hcat, w[layer, :]) 
    end
    
    # x-update can be done in one batch
    update_all_x!(nn, testdata1)
end
