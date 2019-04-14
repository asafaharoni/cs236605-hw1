import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula. \\DONE
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        correct_label_scores = x_scores[torch.arange(x_scores.shape[0]), y]
        # aux_matrix = self.delta * torch.ones_like(x_scores, dtype=torch.float32) + x_scores
        aux_matrix = self.delta + x_scores - correct_label_scores.view(-1,1)
        aux_matrix[aux_matrix<0] = 0
        aux_matrix[torch.arange(aux_matrix.shape[0]),y] = 0
        loss = torch.mean(torch.sum(aux_matrix, dim=1))


        # TODO: Save what you need for gradient calculation in self.grad_ctx  \\DONE

        self.grad_ctx['M'] = aux_matrix
        self.grad_ctx['y'] = y
        self.grad_ctx['x'] = x

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        y = self.grad_ctx['y']
        x = self.grad_ctx['x']
        M = self.grad_ctx['M']
        G = torch.zeros_like(self.grad_ctx['M'])
        G[M > 0] = 1
        G[torch.arange(G.shape[0]), y] = -torch.sum(G, dim=1)
        grad = torch.mm(x.t(), G) / x.shape[0]
        # ========================
        return grad
