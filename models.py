import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        dot = nn.DotProduct(x, self.get_weights())
        return dot

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        scalar = nn.as_scalar(self.run(x))
        if scalar >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            # iterate through dataset once + apply perceptron updates
            # break if it's converged (otherwise, next loop)
            it = dataset.iterate_once(1)
            conv = True
            for x,y in it:
                if self.get_prediction(x) != nn.as_scalar(y):
                    conv = False
                    self.get_weights().update(x, nn.as_scalar(y))
            if conv:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # from piazza question, "where do we set the hidden layer size"
        self.d = 1
        # of features per datapoint
        self.h = 20
        # hidden layer size
        self.hn = 1
        # number of hidden layers
        self.batch_size = 10
        # size of batch, to be chosen
        self.x = nn.Parameter(1, self.d)
        # batch of dim batch_size x d
        self.w1 = nn.Parameter(self.d, self.h)
        self.w2 = nn.Parameter(self.h, 1)
        self.b1 = nn.Parameter(1, self.h)
        self.b2 = nn.Parameter(1, 1)

        self.multiplier = -.001

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w1)
        xw1b1 = nn.AddBias(xw1, self.b1)
        reluxw1b1 = nn.ReLU(xw1b1)
        reluxw1b1w2 = nn.Linear(reluxw1b1, self.w2)
        reluxw1b1w2b2 = nn.AddBias(reluxw1b1w2, self.b2)
        return reluxw1b1w2b2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for i in range(100):
                it_0 = dataset.iterate_once(self.batch_size)
                for x, y in it_0:
                    grad_wrt_w2, grad_wrt_w1, grad_wrt_b2, grad_wrt_b1 = nn.gradients(self.get_loss(x,y), [self.w2, self.w1, self.b2, self.b1])
                    self.w2.update(grad_wrt_w2, self.multiplier)
                    self.w1.update(grad_wrt_w1, self.multiplier)
                    self.b2.update(grad_wrt_b2, self.multiplier)
                    self.b1.update(grad_wrt_b1, self.multiplier)
            it_1 = dataset.iterate_once(self.batch_size)
            avg = 0
            num = 0
            for x, y in it_1:
                avg += nn.as_scalar(self.get_loss(x, y))
                num += 1
            avg = avg / num
            if avg <= .02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # from piazza question, "where do we set the hidden layer size"
        self.d = 784
        # of features per datapoint
        self.h = 100
        # hidden layer size
        self.hn = 1
        # number of hidden layers
        self.batch_size = 10
        # size of batch, to be chosen
        self.x = nn.Parameter(self.batch_size, self.d)
        # batch of dim batch_size x d
        self.w1 = nn.Parameter(self.d, self.h)
        self.w2 = nn.Parameter(self.h, 500)
        self.w3 = nn.Parameter(500, self.batch_size)

        self.b1 = nn.Parameter(1, self.h)
        self.b2 = nn.Parameter(1, 500)
        self.b3 = nn.Parameter(1, self.batch_size)

        self.multiplier = -.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w1)
        xw1b1 = nn.AddBias(xw1, self.b1)
        reluxw1b1 = nn.ReLU(xw1b1)
        reluxw1b1w2 = nn.Linear(reluxw1b1, self.w2)
        reluxw1b1w2b2 = nn.AddBias(reluxw1b1w2, self.b2)
        reluxw1b1w2b2w3 = nn.ReLU(reluxw1b1w2b2)
        reluxw1b1w2b2w3b3 = nn.Linear(reluxw1b1w2b2w3, self.w3)
        reluxw1b1w2b2w3b3last = nn.AddBias(reluxw1b1w2b2w3b3, self.b3)
        # do like 3 layers with relu, linear, addbias
        return reluxw1b1w2b2w3b3last

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        done = False
        while not done:
            for i in range(100):
                it_0 = dataset.iterate_once(self.batch_size)
                for x, y in it_0:
                    grad_wrt_w3, grad_wrt_w2, grad_wrt_w1, grad_wrt_b3, grad_wrt_b2, grad_wrt_b1 = nn.gradients(self.get_loss(x,y), [self.w3, self.w2, self.w1, self.b3, self.b2, self.b1])
                    self.w3.update(grad_wrt_w3, self.multiplier)
                    self.w2.update(grad_wrt_w2, self.multiplier)
                    self.w1.update(grad_wrt_w1, self.multiplier)
                    self.b3.update(grad_wrt_b3, self.multiplier)
                    self.b2.update(grad_wrt_b2, self.multiplier)
                    self.b1.update(grad_wrt_b1, self.multiplier)
                if dataset.get_validation_accuracy() >= .975:
                    done = True
                    break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # "the hidden size d should be sufficiently large"
        self.d = 47
        # of features per datapoint
        self.h = 250
        # hidden layer size
        self.batch_size = 1 # they'll take care of the word length matching within a batch
        # # size of batch
        # batch of dim batch_size x d
        self.expander = nn.Parameter(self.num_chars, self.batch_size)
        # to expand nodes to the correct size in intermediate steps
        self.shrinker = nn.Parameter(self.h, 5)
        # to shrink to batch_size x 5

        self.w1 = nn.Parameter(self.d, self.h)
        self.w2 = nn.Parameter(self.h, self.h)
        self.end_w1 = nn.Parameter(self.h, 100)
        self.end_w2 = nn.Parameter(100, self.h)

        self.b1 = nn.Parameter(1, self.h)
        self.b2 = nn.Parameter(1, 500)
        self.end_b1 = nn.Parameter(1, 100)
        self.end_b2 = nn.Parameter(1, self.h)
            # self.b3 = nn.Parameter(1, self.batch_size)

        self.multiplier = -.01

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # self.batch_size = len(xs[0].data)
        # self.expander = nn.Parameter(self.num_chars, self.batch_size)
        xw1 = nn.Linear(xs[0], self.w1) # 1x47 * 47x100 == 1x100
        xw1b1 = nn.AddBias(xw1, self.b1) # 1x100 + 1x100
        reluxw1b1 = nn.ReLU(xw1b1)
        last_node = reluxw1b1
        # expanded_node = nn.Linear(self.expander, reluxw1b1)
        for i in range(1, len(xs)):
            # print(i)
            # self.batch_size = len(xs[i].data)
            # self.expander = nn.Parameter(self.num_chars, self.batch_size)
            # expanded_node_added = nn.Add(self.w2, expanded_node)
            hw = nn.Linear(last_node, self.w2)
            loop_xw1 = nn.Linear(xs[i], self.w1)
            loop_xw1b1 = nn.AddBias(loop_xw1, self.b1)
            loop_reluxw1b1 = nn.ReLU(loop_xw1b1)
            hw_plus_loop_reluxw1b1 = nn.Add(hw, loop_reluxw1b1)
            last_node = hw_plus_loop_reluxw1b1
            # expanded_node = nn.Linear(self.expander, loop_reluxw1b1)
        end_xw1 = nn.Linear(last_node, self.end_w1)
        end_xw1b1 = nn.AddBias(end_xw1, self.end_b1) # 1x100 + 1x100
        end_reluxw1b1 = nn.ReLU(end_xw1b1)
        end_reluxw1b1w2 = nn.Linear(end_reluxw1b1, self.end_w2)
        end_reluxw1b1w2b2 = nn.AddBias(end_reluxw1b1w2, self.end_b2) # 1x100 + 1x100
        end_reluxw1b1w2b2last = nn.ReLU(end_reluxw1b1w2b2)
        shrunkyclunk = nn.Linear(end_reluxw1b1w2b2last, self.shrinker)
        return shrunkyclunk

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        done = False
        while not done:
            for i in range(100):
                it_0 = dataset.iterate_once(self.batch_size)
                for x, y in it_0:
                    grad_wrt_end_w2, grad_wrt_end_w1, grad_wrt_w2, grad_wrt_w1, grad_wrt_end_b2, grad_wrt_end_b1, grad_wrt_b2, grad_wrt_b1 = nn.gradients(self.get_loss(x,y), [self.end_w2, self.end_w1, self.w2, self.w1, self.end_b2, self.end_b1, self.b2, self.b1])
                    self.end_w2.update(grad_wrt_end_w2, self.multiplier)
                    self.end_w1.update(grad_wrt_end_w1, self.multiplier)
                    self.w2.update(grad_wrt_w2, self.multiplier)
                    self.w1.update(grad_wrt_w1, self.multiplier)
                    self.end_b2.update(grad_wrt_end_b2, self.multiplier)
                    self.end_b1.update(grad_wrt_end_b1, self.multiplier)
                    self.b2.update(grad_wrt_b2, self.multiplier)
                    self.b1.update(grad_wrt_b1, self.multiplier)
                if dataset.get_validation_accuracy() >= .825:
                    done = True
                    break
