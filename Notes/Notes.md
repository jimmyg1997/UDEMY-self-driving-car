# The perceptron

### [ 66 - Overview ]

* **Back propagation**: Find out far too convoluted, complex patters

### [ 71 - Perceptrons]

* Brain does pattern recognition
* *Each node*: includes a linear model inside
* DON'T forget to include the *bias* = 1
* **Activation function**:  It takes the linear combination of these inputs, and check the result
  * <u>Step function</u>: returns yes/no
  * <u>Sigmoid function</u> : returns likeliness

![image-20190509172249646](images/image-20190509172249646.png)

### [ Cross Entropy ]

* *Higher prob*. in points that should be in the **positive** and *lower prob* to the points that should be in the **negative**
* **Cross Entropy** = Sum of logarithms = $- \frac{1}{N}\sum (yln(p) + (1-y)ln(1-p))$



### [ Gradient Descent ]

* $[w_1, w_2, b] - \nabla E $, in python we **can not simply take the derivative of the error function** 

* $ \nabla E = \frac{pts * (p-y)}{m} = \frac{1}{N} \sum_{x}x_(a(x) - y)$

* Substraction the above value from our linear parameters results in new updated weigths for a new line, **but taking small steps in each iteration**, so we need the **learning rate**

---

# Keras

### [ 90 - Keras Models]

* Using Adam optimizer, which is a **stochastic gradient descent**, since original gradient descent is expensive since each time we should calculate mul of e.g. 1 mil. Points
* Adam optimizer also ultimately computes adaptive learning rates for each parameter. It is the best for <u>large models</u> & <u>data sets</u> 

---

# DNN

### [ 94 - Overview ]

![image-20190509205619705](images/image-20190509205619705.png)

### [ 95 - Non - linear Boundaries]

* Treat each linear model as an input node which contains some linear equation

  ![image-20190509205931880](images/image-20190509205931880.png)

### [ 96 - Architecture ]

* [ **$\uparrow$ # h. layers when complex data** ] Use as many **hidden linear layers** as necessary to produce the **non - linearity** needed to fit our model

### [ 97 - Feed - forward process]

* [**Really important for CNN**] **Feed forward** : Get input, produce output & prediction

* **Activation Function**: ReLu

  ![image-20190510025834836](images/image-20190510025834836.png)

### [ 98 - Error Function ]

* The lines are adjusting at specific **learning rates**

* We use the total error to back propagate to update the weights

* <https://playground.tensorflow.org/>

  ![image-20190510032611888](images/image-20190510032611888.png)

  <u>Note</u>: Think 4 linear models as 4 neurons of the hidden layer **eg**. The model wont be trained appropriately if you use 2.



---

# Multiclass Classification

### [ 104 - Softmax ]

* In multiclass we **replace** activation function of *sigmoid* $S(x) = \frac{e^{x}}{1+e^x}$, so we use *softmax*  
  1. The relative mangnitudes of the scores must be maintained! $P_2[.] > P_1[.] > P_0[.]$
  2. $\sum_{i}P_i[.] = P_2[.] + P_1[.] + P_0[.] = 1$

* **Softmax**: $P(score m) = e^{m} / \sum_{i = 1}^{n}e^{i}$

* **Labeling encoding**: with 0, 1, 2 assumes dependencies between the classes, the algorithm could consider it as an order (favouring 2 from 1): So we use **one hot encoding** (*linearly independent*)

  ![image-20190511203647024](images/image-20190511203647024.png)

### [ 105 - Cross Entropy]

* Let's see why **one hot encoding is useful**:  

  ![image-20190511211034657](images/image-20190511211034657.png)

* **Cross Entropy** = $- \sum_{i = 1}^{n} \sum_{j=1}^{m}ln(p_{ij})$ is affected the rest Gradient Descent & Back propagation are not!



---

# MNSIT Image Recongnition

### [ MNIST Datasets ]

* **Input**: It must contain 784 nodes of inputs

  ![image-20190511215813451](images/image-20190511215813451.png)

* **Network**:

  ![image-20190511215949232](images/image-20190511215949232.png)



### [ 111 - Train & Test ]

* Instead of training to **look patters (or general features)** since the purpose of ML is generalization,. they just memorize according to their labels

  ![image-20190511221327475](images/image-20190511221327475.png)

  We need:

  1. Small train error (**underfitting**)

     Not able to capture data's trend & fit the train sets.

  2. Small gap test & train error (**overfitting**)

     $\uparrow$ Hidden Layers $\Rightarrow$ $\uparrow$ Memorize 

     <u>Note</u>: Think that increasing the # of layers, you increase the linearities and therefore occurs a non linearity more complex, that fits perfectly to train data, so the model memories features and DOESN't have good accuracy in test data



     ![image-20190511221733128](images/image-20190511221733128.png)

     How to reduce overfitting? ()

     - $\downarrow$ depth & complexity (during validation stage)
     - $\downarrow$ epochs
     - Dim. $\downarrow$  (during validation stage)
     - Regularization
     - Bayesian probability
     - VC dim  (during validation stage)
     - Dropout

     ![image-20190511223532470](images/image-20190511223532470.png)



### [ 112 - Hyperparameters]

* Goals:

  * <u>Train stage</u>: Minimize the training error by tuning parameters, so tuning hyperparameters will maximize the model capacity, SO NO it is not because we don't avoid overfitting
    * **Parameters** = [bias, weights]
  * <u>Validation stage</u>: Minimize the generalization error by tuning hyperparameters and keep the best model 
    * **Hyperparameters** = *Quantitative features of a Machine Learning Algorithm that are not directly optimized by minimizing training or in-sample losses such MSE train.*
    * **Which are?**
      * Degree of a polynomial regression
      * Regularization parameter $\lambda$
      * No. Layers  & no. hidden layers
      * Learning rate
    * **Tuning**  = Configure complexity, capacity of the model
    * **Validation set** = Data to tune specifiy hyper parameters in order to capture the trends just enough so that it can generalize to new data

  * <u>Test stage:</u> 



---

# Convolutional Neural Networks

### [ 120 - Convolutions & MNIST]

* Understand the **spatial structure** of the inmputs are relevant, they process data with known **grid topology**
* **Pooling** : Continuously reduce the no. of parameters in computations in the network

### [ 121 - Convolution Layer ]

![image-20190512053059804](images/image-20190512053059804.png)		

* **Convolution Layer**: Extract & learn specific image features

  * Use **kernel matrix** (filter) to stride, parsing the image in receptive fields, multiplying the corresponding cells and divide by 9 (if 3x3 kernel), finally creating the feature map. <u>After</u> using convolution layer, we use **relu** activation function to the feature map

    ![image-20190512053647168](images/image-20190512053647168.png)

    * **Edge detection**:

      ![image-20190512053752133](images/image-20190512053752133.png)

    * **Sobel Edge Operator**:

      ![image-20190512053829824](images/image-20190512053829824.png)

    * **Laplacian operator**:

      ![image-20190512053850970](images/image-20190512053850970.png)

  * **Translational Invariance**: The kernel find a feature, it may detect the same somewhere else, NO matter where it is.

  * $\uparrow$ Filters $\Rightarrow$ $\uparrow$ Features extracted $\Rightarrow$ $\uparrow$ the ability to recognize patterns. *Lower* levels to simple aspects of the image (**edges**), *higher* for sophisticated features (**shapes**, **patterns**). SO LESS INFO ABOUT THE IMAGE, BUT MORE INFO ABOUT THE FEATURES THAT ARE DISTINCT TO THE KERNEL INVOLVED ON

    * Wherever the object is in the image **does not matter**

  * <u>Example</u>:

    ![image-20190512063259078](/Users/jimmyg1997/Library/Application Support/typora-user-images/image-20190512063259078.png)

    * Use as filters, specific 3 features, leading to a stack of filtered images



### [ 123 - Pooling]

* **Pooling** : $\downarrow$ the dim of each feature map $\Rightarrow$ $\downarrow$ the complexity of the model $\Rightarrow$ avoid overfitting because it **provives an generalized abstract form of the original feature map**, preserving the patters! **[max, sum, average]**

* **max**: Provides scale invariant representation, so detect features no matter where they are. So you can generalize! eg. for all x's

  ![image-20190512064620902](images/image-20190512064620902.png)



### [ 124 - Fully 	Conected Layer ]

* <u>Train Stage</u>:
  * Convolution: Just the filter values
  * Fully connected: Weights, biases

### [ 126 - Code I ]

* **LeNet**: 1990 as convolutional architecture [Alex, Zepth, Google]

  ![image-20190512152029850](images/image-20190512152029850.png)

* **Padding** = to extract low level features, keep info in the border $\Rightarrow$ $\uparrow$ performance

* **Dropout**:

  * Randomly sets nodes turned off, at each update during training, forces the network to use various combiantions of nodes to classify the same data, forces to learn in more independent way [uniformly distributed]
  * Between <u>convolutional layers</u> or between <u>fully connected layers</u>, generally between layers with **high no. of parameters**, theses are more likely to overfit & memorize data

---

# Classifying Road Symbols

### [ 137 - Fit Generator]

![image-20190513225941244](images/image-20190513225941244.png)



---

# Polynomial Regression

*Train a model that can predict steering angles based on a continuous spectrum, so fit it in a 2D continuum status*

![image-20190514040618732](images/image-20190514040618732.png)



---

# Behaviour Cloning

### [ 144 - Overview ]

*Cars are typically driven around and trained on real roads by manual drivers and then they are trained on the data & clone the behavior of manual drivers*

1. Use simulator to take images in the movement of the car
   * $X_{train} = snapshot, y_{train} = steering angle$

2. Input the $X_{train}$ into a CNN network in order to train model to adjust the steering angle

### [ 145 - Collecting data ]

* The 3 cameras collect data for 
  * **steering angle**
  * **speed** 
  * **trhottle**
  * **break**
* Not a **classification problem** but a **regression problem**, since we try to predict steering angle based on a continuous spectrum!



### [ 146 - Downloading data ]

* ![image-20190520235354711](images/image-20190520235354711.png)
  * If we leave it like this it is biased to pich zero steering at each time! We will decline samples above specific threshold

### [ 150 - Defining NVIDIA model ]

![image-20190521215451218](images/image-20190521215451218.png)

![image-20190521215517926](images/image-20190521215517926.png)



### [ 152 - Flash & Socket.io ]

![image-20190521225302195](images/image-20190521225302195.png)



---

# Extra Notes upon Performance [ Check /src]

1. <u>Multiclass Classification of images</u>: 

   - We use **relu** activation function for hidden layer nodes & softmax for the **output** layer

   - **CNN**: Think an image 72x72 pixels RGB (3 channels), so the input would be 3 * 72 * 72 = 15552 nodes, which cannot been trained with classical DNN (no scalable)

2. <u>CNN</u>:

   * <u>Before th Dropout layer [30 epochs]</u>

     ![image-20190512161156391](images/image-20190512161156391.png)

   * <u>After adding Dropout layer [30 epochs]</u>

     ![image-20190512162335949](images/image-20190512162335949.png)

3. <u>CNN road symbols</u>:

   * **30 epochs, leNet Model**

     ![image-20190513210007753](images/image-20190513210007753.png)

     **PROBLEMS**:

      	1. Low accuracy
      	2. Overfitting

     **SOLUTIONS**

     1. Atom **optimizer** computers individual learning rates, we set an initial! When having complex data it is cood to have small lr:

        * lr = 0.01 => lr = 0.001

          ![image-20190513210626160](images/image-20190513210626160.png)

        * But we still have overfitting as it seems in the second diagram, **increase the no. of filters** inside the convolutional layer (x2 in each layer)

          ![image-20190513211155168](images/image-20190513211155168.png)

     * Increase the **no. of convolutional layers** 2 more layers. BUT THE TOTAL no. Of parameters decreased from 580.000 to 378.023, because with each convolutional layer the dimensions of the image decrease

       ![image-20190513211639550](images/image-20190513211639550.png)

     * Use more than one **dropout layer is common and can be very effective technique**. Add another before the Flatten() layers. The no. of parameters they stay the same

       ![image-20190513212101249](images/image-20190513212101249.png)

     * **Image generator**: The huge gap is due to the many dropout layers!

       ![image-20190513223253002](images/image-20190513223253002.png)

     * **Image generator**: Use of just 1 layer

       ![image-20190513223924716](images/image-20190513223924716.png)


4. <u>Self Driving car</u>

   * **1st attempt:**

     ![image-20190521223056808](images/image-20190521223056808.png)

   *  It appears it has to do with the choice of our activation function*. We had replaced sigmoid activation function with relu to avoid vanishing gradient problem in complex NN*. HOWEVER it caused **dead relu** 

     * **When a node dies and only feeds a gradient value of 0 to the nodes!**

     * The derivative of relu is 1 or 0 for negative values! Since back propagation uses this gradient value to change the weight, now it doesnt, so the model is not trained!

     * Use of **elu activaton function**

       ![image-20190521223654905](images/image-20190521223654905.png)

   ![image-20190521224110080](images/image-20190521224110080.png)

   * Add **2 more dropout layers**, change back to 30 epochs

     ![image-20190521224458366](images/image-20190521224458366.png)

   * Using **augmented** data

     ![image-20190523005203825](images/image-20190523005203825.png)

   * Remove the 2 dropout layers, and PUT ALSO LEFT AD RIGHT IMAGES LOL (<u>they were not in the train data until know, been forgotten</u>)

     ![image-20190523015101599](images/image-20190523025632053.png)
   * **increase** in 15 epochs

     ![image-20190523142303637](images/image-20190523142303637.png)
