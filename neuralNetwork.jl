
module neuralNet

export network, initialize_w!, sigmoid, sigmoid_gradient, sigmoid_hessian, 
update_single_x, update_all_x!, evaluate, evaluate_all_relax, 
evaluate_all_orig, hessian, w_update!

"""
A type for neural network

# Arguments
* `archi`: An array of integers specifying how many neurons are in each hidden layer. 

# Examples
```julia
julia> network([1000, 500, 1000])
1
```
"""
type network
    
    # number of layers
    n:: Int64
    
    # number of neurons in each layer
    size:: Array{Int64, 2}
    
    # weight matrices in each layer
    X:: Array{Array{Float64, 2}, 1}
    
    # offset/bias in each layer
    b:: Array{Array{Float64, 1}, 1}
    
    # data variables
    w:: Array{Array{Float64, 2}, 1}
    
    # sX = t(X) * X
    sX:: Array{Array{Float64, 2}, 1}
    
    # mu: penalty coef for each layer
    μ:: Array{Float64, 1}
    
    # customized constructor
    # args:
    #      archi: # of neurons per layer
    function network(archi)
        
        this = new();
        this.size = archi;
        this.n = size(archi)[2];
        this.μ = ones(this.n)
        
        # random initialize weights
        this.X = Array{Array{Float64, 2}, 1}()
        for i = 1:(this.n - 1)
            push!(this.X, (rand(this.size[i + 1], this.size[i]) .- 0.5) * 2 * 4 / sqrt((this.size[i + 1] + this.size[i]) / 6));
        end
    
        # t(X) * X
        this.sX = Array{Array{Float64, 2}, 1}()
        for i = 1:(this.n - 1)
            push!(this.sX, this.X[i]' * this.X[i]);
        end
        push!(this.sX, eye(this.size[end])); 
        
        # offset initialization
        this.b = Array{Array{Float64, 1}, 1}()
        for i = 1:(this.n - 1)
            push!(this.b, (rand((this.size[i + 1])) .- 0.5) * 2 * 4 / sqrt((this.size[i + 1] + this.size[i]) / 6));
        end
        
        return this
    end
end



"""
Initialize the W-variables for a neural net

# Arguments
* `nn`: A network object. 
* `traindata`: training data.
"""
function initialize_w!(nn, traindata)
    
    # m := number of data
    m = size(traindata, 2)
    
    # initialize w
    nn.w = Array{Array{Float64, 2}, 1}()
    
    # input layer
    push!(nn.w, nn.X[1] * traindata + nn.b[1] * ones(1, m))
    
    # compute each layer
    for i = 2:(nn.n - 1)
         push!(nn.w, nn.X[i] * (ones(nn.size[i], m) ./ (1 + exp(-nn.w[i - 1]))) + nn.b[i] * ones(1, m));
    end
end



"""
Calculate the sigmoid of an array of numbers

# Arguments
* `w`: An array.
"""
sigmoid(w) = ones(size(w)) ./ (1 + exp(-w))



"""
Calculate the gradient of the sigmoid of an array of numbers

# Arguments
* `w`: An array.
"""
sigmoid_gradient(w) = sigmoid(w) .* (1 - sigmoid(w))



"""
Calculate the hessian of sigmoid function

# Arguments
* `w`: An array.
"""
sigmoid_hessian(w) = sigmoid(w) .* ((1 - sigmoid(w)) .^ 2) - (sigmoid(w) .^ 2) .* (1 - sigmoid(w))



"""
Update the weights (X) and bias (b) of a single layer

# Arguments
* `μ`: penalty coefficient.
* `ϕ`: sigmoid of W's.
"""
function update_single_x(μ, ϕ, new_w)
    
    # solve OLS
    ϕ_aug = vcat(ϕ, ones(1, size(ϕ)[2]))
    X_b = (new_w * ϕ_aug') * pinv(ϕ_aug * ϕ_aug' + 1 / 6 / μ  * eye(size(ϕ_aug, 1)))
    X = X_b[:, 1:end - 1];
    b = X_b[:, end];
    return X, b
end



"""
Update the weights and biases for all layers

# Arguments
* `nn`: a network object.
* `v`: input to the neural net.
"""
function update_all_x!(nn, v)
    
    ϕ = Array{Array{Float64, 2}, 1}()
    push!(ϕ, v)

    # update ϕ for each layer
    for i = 2:(nn.n - 1)
        push!(ϕ, ones(size(nn.w[i - 1])) ./ (1 + exp(-nn.w[i - 1])))
    end

    # put new weights and bias into the neural network
    for i = 1:(nn.n - 1)      
        nn.X[i], nn.b[i] = update_single_x(nn.μ[i], ϕ[i], nn.w[i]);
    end
end



"""
Compute function value for one data point

# Arguments
* `nn`: a network object.
* `v`: input to the neural net.
* `y`: actual output.
"""
function evaluate(nn, v, y)
    
    # last layer
    f = 1 / 2 * norm(sigmoid(nn.w[end]) - y, 2)^2;
    
    # hidden layers 
    for i = (nn.n - 1) : -1 : 2
        f = f + 1 / 2 * nn.μ[i] * norm(nn.X[i] * sigmoid(nn.w[i - 1]) + nn.b[i] - nn.w[i], 2) .^ 2;
    end
    
    # first layer
    f = f + 1/2 * nn.μ[1]* norm(nn.X[1] * v + nn.b[1] -nn.w[1], 2) .^ 2;
    return f
end



"""
Evaluate the relaxed problem

# Arguments
* `nn`: a network object.
* `v`: input to the neural net.
* `y`: actual output.
"""
function evaluate_all_relax(nn, v, y)
    
    m = size(nn.w[1], 2);
    n = nn.n - 1;
    ff = 1/2 * sum(sum((sigmoid(nn.w[n]) - y) .* (sigmoid(nn.w[n]) - y)));
    infeasibility = 1/2 * nn.μ[1]* norm(nn.X[1] * v + nn.b[1]* ones(1, m) - nn.w[1], 2) ^ 2;
    for i = 2 : -1 : nn.n-1
        infeasibility = infeasibility + 1/2 * nn.μ[i] * norm(nn.X[i] * sigmoid(nn.w[i - 1]) + 
                        nn.b[i] * ones(1, m) - nn.w[i], 2) ^ 2;
    end
    ff = ff + infeasibility;
    return ff, infeasibility
end



"""
Evaluate the objective function for the encoder-decoder problem, 
Error is defined as 0.5 * norm(input - output).

# Arguments
* `nn`: a network object.
* `v`: input to the neural net.
* `y`: actual output.
"""
function evaluate_all_orig(nn, v, y)
    m = size(nn.w[1], 2);
    result = nn.X[1] * v + nn.b[1] * ones(1, m);
    for i = 1:nn.n -2
        result = ( nn.X[i+1] * sigmoid(result) + nn.b[i + 1] * ones(1, m));
    end
    difference = sigmoid(result) - y;
    1 / 2 * sum(difference .^ 2);
end



"""
Calculate gradient and hessian of W's

# Arguments
* `nn`: a network object.
* `v`: input to the neural net.
* `y`: actual output.
"""
function hessian(nn, v, y)
    
    # s: the cumulative sizes since the second layer
    s = hcat(0, cumsum(nn.size[2:end], 1)')
    
    # ϕ := ϕ(w) function value after applying the sigmoid
    ϕ = Array{Array{Float64, 2}, 1}(nn.n - 1);
    
    # its gradient
    ϕ_gradient = Array{Array{Float64, 2}, 1}(nn.n - 1);
    
    # for each layer: fill in ϕ and its gradient
    for i = 1:nn.n - 1
        ϕ[i] = sigmoid(nn.w[i]);
        ϕ_gradient[i] = sigmoid_gradient(nn.w[i]);
    end
    
    # if more than two layers (i.e. existence of hidden layers)
    if nn.n > 2
        dif = Array{Array{Float64, 2}, 1}(nn.n)
        for i = 1:nn.n - 2
            dif[i + 1] = nn.sX[i + 1] * ϕ[i] - nn.X[i + 1]' * (nn.w[i + 1] - nn.b[i + 1]);
        end
    end
    
    i = nn.n - 1;
    dif[i + 1] = ϕ[i] - y;
    
    # H: the Hessian
    H = zeros(s[nn.n], s[nn.n]);
    for i = 1:nn.n - 1
        H[s[i] + 1:s[i + 1], s[i] + 1:s[i + 1]] = nn.μ[i + 1] * nn.sX[i + 1] .* (ϕ_gradient[i] * ϕ_gradient[i]') + 
                                                nn.μ[i + 1] * diagm(vec(dif[i + 1] .* sigmoid_hessian(nn.w[i]))) + 
                                                nn.μ[i] * eye(nn.size[i + 1]);
    end
    for i = 1:nn.n - 2
        H[s[i] + 1:s[i + 1], s[i + 1] + 1:s[i + 2]] = -nn.μ[i + 1] * nn.X[i + 1]' .* (ϕ_gradient[i] * ones(1, nn.size[i + 2]));
        H[s[i + 1] + 1:s[i + 2], s[i] + 1:s[i + 1]] = H[s[i] + 1:s[i + 1], s[i + 1] + 1:s[i + 2]]';
    end
    
    # g: the gradient
    g = zeros(s[nn.n], 1);
    i = 1;
    g[s[i] + 1:s[i + 1]] = nn.μ[i + 1] * dif[i + 1] .* ϕ_gradient[i] - nn.μ[i] * (nn.X[i] * v + nn.b[i] - nn.w[i]);
    for i = 2:nn.n - 1
        g[s[i] + 1:s[i + 1]] = nn.μ[i + 1] * dif[i + 1] .* ϕ_gradient[i] - nn.μ[i] * (nn.X[i] * ϕ[i - 1] + nn.b[i] - nn.w[i]);
    end
    
    return H, g
end



"""
Update W's 

# Arguments
* `nn`: a network object.
* `v`: input to the neural net.
* `y`: actual output.
* `iter`: number of iterations.
"""
function w_update!(nnw, v, y, iter)
    
    # α and β are parameter to do damped Newton
    α = 0.2;
    β = 0.25;
    tor = 0.0001;
    
    cumu_size = hcat(0, cumsum(nnw.size[2:end], 1)')
    
    for k = 1:iter
        
        # H, g are computed Hessian and gradient
        H, g = hessian(nnw, v, y)
        direction = H \ g
        
        # # Matlab code for gradient and Hessian in case Julia blows up
        # LA, DA = mxcall(:ldl,2,H);
        # if minimum(diag(DA)) > tor
        #     direction = LA' \ (DA \ (LA \g));
        # else
        #     tmp = diag(DA);
        #     tmp[find(abs(tmp) .< tor)] = tor * ones(size(find(abs(tmp) .< tor)));
        #     tmp[find(tmp .< -tor)] = - tmp[find(tmp .< -tor)];
        #     direction = LA'\(diagm(tmp)\(LA\g));
        # end

        # do backtracking line search
        t = 1;
        
        objective_value = evaluate(nnw, v, y);
        
        # need to avoid shallow copy
        temp_nnw = deepcopy(nnw);
        
        # take a step of size t for each layer of W's
        for i = 1:nnw.n - 1
            temp_nnw.w[i] = deepcopy(nnw.w[i]) - t * direction[cumu_size[i] + 1:cumu_size[i + 1]];
        end
        
        # value of objective function given the current step size
        current = evaluate(temp_nnw, v, y);
        
        while  current > (objective_value - α * t * g' * direction)[1, 1]
            
            # shrink the step size by a coefficient of β every time
            t = β * t;
            for i = 1:nnw.n - 1
                temp_nnw.w[i] = deepcopy(nnw.w[i]) - t * direction[cumu_size[i] + 1:cumu_size[i + 1]];
            end
            current = evaluate(temp_nnw, v, y);
        end
        
        # update w
        for i = 1:nnw.n - 1
            nnw.w[i] -= t * direction[cumu_size[i] + 1:cumu_size[i + 1]];
        end 
    end
end

end
