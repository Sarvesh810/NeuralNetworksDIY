
# Unboxing Neural Networks: 📦 -> 🔢✖️🔢
I have been playing with neural networks for quite a while building dog-cat classifiers, predicting housing prices and classifying handwritten digits (my first ever experience with neural networks) and I was always fascinated by how could a network learn. Until now, I've been randomly tweaking the networks I had created by adding layers, changing activation functions, changing loss functions or adding more neurons to layers in hopes that something clicks somewhere that improves the accuracy of predictions (I didn't know of any other metrics either until I came across the confusion matrix in my CS with AI course). Neural Networks were like a black box to me in which I put in some data and it would magically spit out results but when things didn't work well, I had this frustration of not knowing how to improve upon the network.

In this project, I'll be looking inside the black box to see how things work, especially how networks learn (which intrigues me the most). I'll do this by creating a neural network from the ground up only using numpy library.

### What actually is inside the black box?
I was surprised to find out that a neural network, as a whole, is a very complex math function or rather a chain of functions. A neural network can be divided up into  3 parts:
1. **Neurons -** mini functions that take in input from previous layers and sum the product of their corresponding '***weights***' and add a '***bias***' term.
2. **Activation functions -** provide non-linearity to the network (otherwise a neuron's output would simply be ***y=mx+b*** which is a straight line).
3. **Loss function -** calculates how wrong the output of the network is from the '***true value***.' 

### What data can it be trained on?
I want the design to be modular so it's not dependent on any particular dataset to be tested on. Although I have used the MNIST hand-written digits dataset to test the neural net. This is partly because I would further want to dive deeper into **O**ptical **C**haracter **R**ecognition (OCR).

### How to use the repo?
Open the command-line interface of your choice
1. Install numpy (if you haven't already)<br>
	`pip install numpy`
 
2. Clone the repository in your desired folder<br>
	`git clone https://github.com/Sarvesh810/NeuralNetworksDIY.git`

3. Check the example code<br>
	[Neural Network Example](https://github.com/Sarvesh810/NeuralNetworksDIY/blob/main/notebooks/mnist_example_notebook.ipynb)

### Optional Reading
- [How Neural Networks Learn](https://khandelwalsarvesh.hashnode.dev/unboxing-neural-networks-a-quick-summary)
