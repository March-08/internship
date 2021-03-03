### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ a7bf9be3-2082-44d0-bdfa-db3da7f200e9
begin
	using Pkg;
#	Pkg.add(Pkg.PackageSpec(;name="Flux", version="0.10"))
	Pkg.add("Flux");
	Pkg.add("PlutoUI");
	using Flux;
	using PlutoUI;
end

# ╔═╡ 62ed3b30-7b95-11eb-2e1b-71800b0f5b0b
begin
	#Pkg.rm("Plots") 
	Pkg.gc()
	Pkg.add("Plots");	
	using Plots
end

# ╔═╡ f6e57470-7aa4-11eb-3ab6-3d2987decbb0
using Flux: onehotbatch, onecold


# ╔═╡ 5e198d96-99a5-4270-8999-135f3b062798
using Flux.Data: DataLoader

# ╔═╡ 714063a7-5530-4f13-ac71-daafc82dd3d7
using Statistics #to have mean function

# ╔═╡ fbea00b2-822a-41e9-8b3a-9c8346d04cde
using Flux: @epochs,throttle

# ╔═╡ ac467bd2-8bad-4f34-988b-778208d13a87
md"""
### Let's use MNIST data!

First of all we can use mnist training and test data in the following way.
The images are gray scale , 28x28 pixels, and each image has a label which indicate the number, we can visualize the i-th image and label by writing images[ i ] and labels[ i ]
"""

# ╔═╡ 6be25dc9-3b32-4790-853c-2211793b1b8b
begin
	images = Flux.Data.MNIST.images()[1:2000];
	labels = Flux.Data.MNIST.labels()[1:2000]
	
	images_test = Flux.Data.MNIST.images(:test)[1:2000]
	labels_test = Flux.Data.MNIST.labels(:test)[1:2000]
	
	print("label : " ,labels[1], "\nsize : ", size(images[1]))
	images[1]
end

# ╔═╡ 90c29fd0-7b56-11eb-2a1e-43c782ffa4c6
md"CI SONO SOLO LE PRIME 100 IMMAGINI --> DA SISTEMARE"

# ╔═╡ dd0a0241-5e36-49c6-81ae-24bef08974ea
md"""


### Lets plot the distribution of our data
so the number of 0, 1 , 2 ...9  that we have in our training set and test set. I use the plot() function to combine the two histograms in a single plot
"""

# ╔═╡ 4e774222-869e-4700-94a7-d095e8522e7d
begin	
	p1 = histogram(labels, xticks=(0:1:9) , label = "train data") 
	p2 = histogram(labels_test, xticks=(0:1:9), color = :green , label = "test data")
	
	plot(p1, p2, layout = (2, 1), legend = true)	
end

# ╔═╡ 0b3970b3-1a08-49f9-9f38-efe6061b5c8e
md"""
### We need to reshape our data

onehotbatch takes a batch of labels end convert it in [one hot vectors](https://en.wikipedia.org/wiki/One-hot), ex: the conversion of 3 is (0,0,1,0,0,0,0,0,0) becasue the third entry of the vector is 1. We need this conversion because it is easier to work with this representation.
Note that the **onecold** function is the inverse, from the one-hot we return to the original number



"""

# ╔═╡ 79c94fa0-5129-4698-9b43-d91cd058f187
begin	
	X = hcat(float.(reshape.(images, :))...) #stack all the images
	Y = onehotbatch(labels, 0:9) # just a common way to encode categorical variables
end

# ╔═╡ 92379275-a5f7-4aa9-adba-de3de2af17cc
md"""
I now use the DataLoader functionality to aggregate the X and Y data in a single site, and divide them in batches by using 
the batchsize parameter.
You can think a [batch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) as a "group" of input data that is given to the Neural Network to start its computation.
When all the batches have been eaten by our Neural Network we can say that the fisrt [epoch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) is terminated, and we can start
again with the second epoch
"""

# ╔═╡ f0626ea0-7aa4-11eb-293b-dd3d023b7e07
#data = DataLoader((X,Y),batchsize= 128)       # or data = zip(X, Y);
data = zip(X,Y)

# ╔═╡ b1c112ac-19f3-45d0-a74c-f457ce0a91a5
md"""
### Define the model

Each input is an image of **28x28** = 784 pixels, so we need a Neural Net. that can allow 784 numbers in input. Note that each pixel is just a number from 0 to 1 that indicates the quantity of black in it. 0 is a white pixel and 1 is a black one.

We are going to use the **RELU** activation functions as it has been demostrated to do well on this task.
The output is a layer with 10 units, and to interpret the output as a probability distribution I use a **softmax** function in the output and of course a **crossentropy** loss function.
![The network](https://lh3.googleusercontent.com/proxy/dVO6UbU2uh1SNUNYwF12QHGCoMnm4qN8TJC6T89pMOC7VrZzd2E6Oiqg9ms7p4yk_IO5pDgwUyOPPW6m8owOJN_fq1u5RtuAoSYHg5YZEHc1nQ)
"""

# ╔═╡ 4a707ef1-be02-4404-b2a6-b513aeaedde5
#use gpu() function to run on gpu

mnist_model = Chain(
  Dense(28^2, 15, relu),
  Dense(15, 10),
  softmax);


# ╔═╡ 9e97ca40-7ab1-11eb-1a26-955cfc509dc4
Flux.params(mnist_model)[1]

# ╔═╡ 2a19de93-23a3-4dae-bd3c-442ad8fe4a7f
md"""
### L2 Regularization

[L2 regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c) technique work by **limiting the capacity** of models such as neural networks or linear regression by adding a parameter of to the objective function (loss function).
Note that we can take all the params W and b in our model simply using Flux.params(model)
![L2 Regularization](https://miro.medium.com/max/631/1*6dfxa-smu8nRZwiFkSwriA.png)
"""

# ╔═╡ 89d91187-24a9-424d-bc14-9da1f53e4f57
begin
	sqnorm(x) = sum(abs2, x); #function that takes square of number x
	λ = 0.001
	#apply square to each param and sum
	regularization_term = (λ * sum(sqnorm, Flux.params(mnist_model)))
	
end

# ╔═╡ 951bcf99-3daa-4cab-9784-7da447b1d8c4
md"""
 Now we can define our objective **loss function** whith the **additional regularization term**.
 In here I also define **Adam** that will be our optimizer, we can also use Descent() for gradient descent etc...
"""

# ╔═╡ 11946eb1-5fce-4821-a52a-6438a809cc40
begin
	loss(x,y) = Flux.crossentropy(mnist_model(x),y) + regularization_term	
	opt = ADAM();
end

# ╔═╡ 51cf3269-53a8-4bd0-bee5-631261111353
md"""
### Let's train!

First of all we need our parameters (weights and biases). Because when we train our model we need to update these value constantly. Flux let us to extract the gradient these parameters from the model with a simple function.
"""

# ╔═╡ d6beedfa-b95f-4494-9259-607df4b4ea0e
ps = Flux.params(mnist_model)

# ╔═╡ 813216d0-7aa8-11eb-23d7-61b5ea39b893
number_weights = lastindex(ps[1])

# ╔═╡ 689bba30-7aa9-11eb-04e7-d140d85e096c
number_biases = lastindex(ps[2])+lastindex(ps[4])

# ╔═╡ 12dd397e-ebd5-44b1-aa42-15c03c2da0a3
md"""
Define an accuracy function to evaluate our model. The accuracy in machine learning is simply defined by : ![image.png](https://lawtomated.com/wp-content/uploads/2019/10/Accuracy_2.png)
"""

# ╔═╡ d753d7f0-7aa9-11eb-29b5-c598de0d2e57
accuracy(x, y) = mean(onecold(mnist_model(x)) .== onecold(y))

# ╔═╡ 2c734cc1-ea31-4bc3-88e0-4fa27f64a32a
md"""
We need @epochs and throttle to run several epochs as mentioned before, and to use a callback function to monitor the training step.
eva
"""

# ╔═╡ 7c036b14-3dc5-49f9-8f2c-e6c580498617
md"""
The **evalcb** is just the function passed to the callback cb which prints the loss and accuracy reached. **Throttle** specify that we want to call the evalcb function every **2 second**.
@epochs n  runs n epochs
"""

# ╔═╡ 447d5050-7ab3-11eb-0f2b-bbdf4648a625
begin
	# callback to show loss and acc
	@show evalcb = () -> println("\n\n loss : " , loss(X, Y), "  accuracy : ", accuracy(X,Y),"\n biases :" , ps[2][1:10] ,"\nweights: " ,ps[1][1:10])
	
	#change 100 params per time
	#for i in 1:1
	 #   Flux.train!(loss, ps, data, opt, cb = throttle(evalcb,5))
	    ##Flux.params(mnist_model)[1] .= delete_params(ps,7839)
	#end
	 
	@epochs 3 Flux.train!(loss, ps, [(X,Y)], opt, cb = throttle(evalcb,1))
	    
end

# ╔═╡ 2c794470-7abb-11eb-2e5c-519ede428365
"""for batch in data
    
    gradient = Flux.gradient(ps) do
      # Remember that inside the loss() is the model
      # `...` syntax is for unpacking data
      training_loss = loss(batch...)
      return training_loss
    end
    
    Flux.update!(opt, ps, gradient)
	println(Flux.params(mnist_model)[1][1:5])
end"""

# ╔═╡ 17ccb156-1cc3-4caa-ae87-1d8518bf784b
Flux.params(mnist_model)[1]

# ╔═╡ 508aee00-7aba-11eb-057d-4b8c9c510380
ps[1][1:5]

# ╔═╡ 16e125f0-7ab7-11eb-31ba-63c4d5668a5a
println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# ╔═╡ 118cf9de-b189-482c-ad36-4645e8fd5847
md"""
#### same preprocessing step for the test set
"""

# ╔═╡ be336d62-f947-4b6e-97ab-adb5ff1cfc75
begin
	# Same preprocessing for test set
	test_X = hcat(float.(reshape.(images_test, :))...)
	test_Y = onehotbatch(labels_test, 0:9);
end

# ╔═╡ 8aaec379-4743-42ef-b636-e0936c3ef1ca
md"""
Lets now classify the label of a random image in the test set.
Not that the **7-th element in the array is almost 1**, the others instead are nearly to 0.
**This means that our classification is 6**, because Julia start indexing from 1
"""

# ╔═╡ 39a8a872-f94c-48c0-a703-2c989b5e6a82
mnist_model(test_X[:,5287]) # Note the 7th index ( corresponding to the digit 6 ) is nearly 1

# ╔═╡ 40196367-a4fa-4ee1-97b1-2ca32e2936f5
md"""
Finally obtain the label from the array with the onecold function and print the actual label and the prediction label
"""

# ╔═╡ a5c08bfb-ac87-4b09-b91d-27541abf7dff
begin
	#metto -1 perche l indexing in julia parte da 1
	println("actual label : " , onecold(test_Y[:,5287])-1)
	println("prediction : ", onecold(mnist_model(test_X[:,5287]))-1)
	images_test[5287]
end

# ╔═╡ Cell order:
# ╠═a7bf9be3-2082-44d0-bdfa-db3da7f200e9
# ╟─ac467bd2-8bad-4f34-988b-778208d13a87
# ╠═6be25dc9-3b32-4790-853c-2211793b1b8b
# ╟─90c29fd0-7b56-11eb-2a1e-43c782ffa4c6
# ╟─dd0a0241-5e36-49c6-81ae-24bef08974ea
# ╠═62ed3b30-7b95-11eb-2e1b-71800b0f5b0b
# ╠═4e774222-869e-4700-94a7-d095e8522e7d
# ╟─0b3970b3-1a08-49f9-9f38-efe6061b5c8e
# ╠═f6e57470-7aa4-11eb-3ab6-3d2987decbb0
# ╠═79c94fa0-5129-4698-9b43-d91cd058f187
# ╟─92379275-a5f7-4aa9-adba-de3de2af17cc
# ╠═5e198d96-99a5-4270-8999-135f3b062798
# ╠═f0626ea0-7aa4-11eb-293b-dd3d023b7e07
# ╟─b1c112ac-19f3-45d0-a74c-f457ce0a91a5
# ╠═4a707ef1-be02-4404-b2a6-b513aeaedde5
# ╠═9e97ca40-7ab1-11eb-1a26-955cfc509dc4
# ╟─2a19de93-23a3-4dae-bd3c-442ad8fe4a7f
# ╠═89d91187-24a9-424d-bc14-9da1f53e4f57
# ╟─951bcf99-3daa-4cab-9784-7da447b1d8c4
# ╠═11946eb1-5fce-4821-a52a-6438a809cc40
# ╟─51cf3269-53a8-4bd0-bee5-631261111353
# ╠═d6beedfa-b95f-4494-9259-607df4b4ea0e
# ╠═813216d0-7aa8-11eb-23d7-61b5ea39b893
# ╠═689bba30-7aa9-11eb-04e7-d140d85e096c
# ╟─12dd397e-ebd5-44b1-aa42-15c03c2da0a3
# ╠═714063a7-5530-4f13-ac71-daafc82dd3d7
# ╠═d753d7f0-7aa9-11eb-29b5-c598de0d2e57
# ╟─2c734cc1-ea31-4bc3-88e0-4fa27f64a32a
# ╠═fbea00b2-822a-41e9-8b3a-9c8346d04cde
# ╟─7c036b14-3dc5-49f9-8f2c-e6c580498617
# ╠═447d5050-7ab3-11eb-0f2b-bbdf4648a625
# ╟─2c794470-7abb-11eb-2e5c-519ede428365
# ╠═17ccb156-1cc3-4caa-ae87-1d8518bf784b
# ╠═508aee00-7aba-11eb-057d-4b8c9c510380
# ╠═16e125f0-7ab7-11eb-31ba-63c4d5668a5a
# ╟─118cf9de-b189-482c-ad36-4645e8fd5847
# ╠═be336d62-f947-4b6e-97ab-adb5ff1cfc75
# ╟─8aaec379-4743-42ef-b636-e0936c3ef1ca
# ╠═39a8a872-f94c-48c0-a703-2c989b5e6a82
# ╟─40196367-a4fa-4ee1-97b1-2ca32e2936f5
# ╠═a5c08bfb-ac87-4b09-b91d-27541abf7dff
