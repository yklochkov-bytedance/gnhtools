import abc
from typing import Any
import torch
from torch import nn


class BaseTask(abc.ABC):
    """An abstract adapter that provides torch-influence with project-specific information
    about how training and test objectives are computed.

    In order to use torch-influence in your project, a subclass of this module should be
    created that implements this module's four abstract methods.
    """

    @abc.abstractmethod
    def train_outputs(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Returns a batch of model outputs (e.g., logits, probabilities) from a batch of data.

        Args:
            model: the model.
            batch: a batch of training data.

        Returns:
            the model outputs produced from the batch.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def train_loss_on_outputs(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        """Returns the **mean**-reduced loss of the model outputs produced from a batch of data.

        Args:
            outputs: a batch of model outputs.
            batch: a batch of training data.

        Returns:
            the loss of the outputs over the batch.

        Note:
            There may be some ambiguity in how to define :meth:`train_outputs()` and
            :meth:`train_loss_on_outputs()`: what point in the forward pass deliniates
            outputs from loss function? For example, in binary classification, the
            outputs can reasonably be taken to be the model logits or normalized probabilities.

            For standard use of influence functions, both choices produce the same behaviour.
            However, if using the Gauss-Newton Hessian approximation for influence functions,
            we require that :meth:`train_loss_on_outputs()` be convex in the model
            outputs.

        See also:
            :class:`CGInfluenceModule`
            :class:`LiSSAInfluenceModule`
        """

        raise NotImplementedError()

    #@abc.abstractmethod
    #def train_regularization(self, params: torch.Tensor) -> torch.Tensor:
    #    """DEPRICATED!
    #    """
    #
    #    raise NotImplementedError()

    def train_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Returns the **mean**-reduced regularized loss of a model over a batch of data.

        This method should not be overridden for most use cases. By default, torch-influence
        takes and expects the overall training loss to be::

            outputs = train_outputs(model, batch)
            loss = train_loss_on_outputs(outputs, batch) + train_regularization(params)

        Args:
            model: the model.
            batch: a batch of training data.

        Returns:
            the training loss over the batch.
        """

        outputs = self.train_outputs(model, batch)
        return self.train_loss_on_outputs(outputs, batch)

    @abc.abstractmethod
    def test_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Returns the **mean**-reduced loss of a model over a batch of data.

        Args:
            model: the model.
            batch: a batch of test data.

        Returns:
            the test loss over the batch.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def batch_size(self, batch: Any) -> int:
        """Returns the amount of datapoints in the batch. Eg, for language models we need to account for attention mask.

        Args:
            batch: a batch of test data.

        Returns:
            the total size of the batch.
        """

        raise NotImplementedError()

    def get_collate_fn(self) -> Any:
        """Return collate function specific to the task. Needs to be redifined for language modeling tasks.

        Args:

        Returns:
            function or None.
        """
        if hasattr(self, "collate_fn"):
            return self.collate_fn
        else:
            return None

    #@abc.abstractmethod
    #def hessian_normalized_logits(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Ad-hoc method for FIM-vector products.  There are two options:

        1. The original FIM is

            \E_{x \sim data} \E_{\hat{y} \sim \pi(x)} \nabla l(\hat{y}, x) \nabla l(\hat{y}, x)^{\top}

        If we use this formulation, this method has to return a batch of losses l(\hat{y}, x), with \hat{y} generated
        inside this function. Target values given in the batch are ignored.

        2. Where there is too many classes, simulating \hat{y} \sim \pi(x) can be too noisy. Therefore, one may want to
        implement Gauss-Newton Hessian. It is only relevant to the case where \pi(x) = \sf(h(x)) and l(., .) is the
        cross entropy loss. In this case, FIM and GNH both equal too

            \E_{x \sim data} J h(x) {Diag(sf(h(x))) - sf(h(x)) sf(h(x))^{\top}} J h(x)

        Here J h(x) denotes the Jacobian of h(x), has size parameter dim X num of logits. If one uses this method, this
        method must return the rensor (matrix applied in each batch slice separately)

            L_h h(x),

        where we denote

            L_h = Diag(rs(x)}) - rs(x) s(x)^{\top},     s(x) = \sf(h(x)),   rs(x) = sqrt(s(x))

            L_h^{\top} L_h = Diag(s(x)}) - s(x) s(x)^{\top} (TODO: CHECK)

        Above we assume that h(x) is a column, otherwise make appropriate rearrangement. Above, L_h is detached.

        Args:
            batch: a batch of test data.

        Returns:
            the total size of the batch.
        """

    #    raise NotImplementedError()
    
# TODO: add hessian_at etc.
