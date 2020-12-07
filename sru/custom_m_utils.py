from torch import nn
import logging
from typing import List, Union
from sru import SRU, SRUCell

logger = logging.getLogger(__name__)


def convert_sru_cell_to_linears(cell: SRUCell):
    if cell.custom_m is not None:
        if cell.custom_m.__class__.__name__ == 'ProjectedLinear':
            # leave as-is
            logger.debug('cell already ProjectedLinear, leaving as-is')
            return
        elif isinstance(cell.custom_m, nn.Sequential):
            if len(cell.custom_m) != 2:
                raise Exception(
                    'custom_m already in use, cannot convert to Linears ' + str(cell.custom_m))
            if not isinstance(cell.custom_m[0], nn.Linear):
                raise Exception(
                    'custom_m already in use, cannot convert to Linears ' + str(cell.custom_m))
            if not isinstance(cell.custom_m[1], nn.Linear):
                raise Exception(
                    'custom_m already in use, cannot convert to Linears ' + str(cell.custom_m))
            # already converted, no need to do anything
            logger.debug('cell already Sequential[Linear, Linear], leaving as-is')
            return
        elif isinstance(cell.custom_m, nn.Linear):
            # nothing to do
            logger.debug('cell already Linear, leaving as-is')
            return
        else:
            raise Exception(
                'custom_m already in use, cannot convert to Linears ' + str(cell.custom_m))
    else:
        # this is mostly to placate mypy:
        if cell.weight is None:
            raise Exception("Shouldn't be here. Logic error in SRU code")
        # if getattr(cell, 'weight_proj', None) is None:
        if cell.weight_proj is None:
            # not using projection, convert to Linear custom_m
            logger.debug('cell not using projection, converting to Linear')
            input_size, output_size = cell.weight.data.size()
            device = cell.weight.device
            cell.custom_m = nn.Linear(input_size, output_size, bias=False).to(device)
            cell.custom_m.weight.data[:] = cell.weight.data.transpose(0, 1)
            cell.weight = None
        else:
            # using projection, convert to Sequential[Linear, Linear]
            logger.debug('cell using projection, converting to Sequential[Linear, Linear]')
            input_size, projection_size = cell.weight_proj.size()
            output_size = cell.weight.size(1)
            device = cell.weight.device
            first = cell.weight_proj.data.transpose(0, 1)
            second = cell.weight.data.transpose(0, 1)
            linear1 = nn.Linear(input_size, projection_size, bias=False).to(device)
            linear2 = nn.Linear(projection_size, output_size, bias=False).to(device)
            linear1.weight.data[:] = first
            linear2.weight.data[:] = second
            cell.custom_m = nn.Sequential(
                linear1, linear2)
            cell.weight = None
            cell.weight_proj = None


def convert_sru_to_linears(sru: SRU):
    """
    Convert sru module to use nn.Linear, in the custom_m.
    You might do this to facilitate:
    - using quantization (which sometimes only works on Linear modules)
    - FLOP pruning (https://arxiv.org/abs/1910.04732)

    custom_m module is used to multiply the inputs by weight matrices, to
    give the U matrix. This is the largest matrix multiplication in SRU,
    and typically dominates the execution time.

    If an sru cell has projection_size more than zero, then the matrix is factorized
    into two. This can give lower latencies, and is also useful when
    using FLOP pruning.

    When converting SRU using this method, custom_m must not already
    being used. If custom_m is already being used in a cell, then an
    Exception will be thrown, since the custom_m
    is needed to store the Linear modules.

    If projection is used, then custom_m will contain
    a Sequential[Linear, Linear], otherwise it will contain a Linear.
    """
    for cell in sru.rnn_lst:
        convert_sru_cell_to_linears(cell)


def create_sru_using_linears(
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        projection_size: int = 0,
        linear_first_layer: bool = True,
        custom_m: None = None,
        **sru_kwargs) -> SRU:
    """
    Create SRU using nn.Linear, in the custom_m.
    You might do this to facilitate:
    - using quantization (which sometimes only works on Linear modules)
    - FLOP pruning (https://arxiv.org/abs/1910.04732)

    custom_m module is used to multiply the inputs by weight matrices, to
    give the U matrix. This is the largest matrix multiplication in SRU,
    and typically dominates the execution time.

    If projection_size is more than zero, then the matrix is factorized
    into two. This can give lower latencies, and is also useful when
    using FLOP pruning.

    When creating SRU using this method, you cannot specify a custom_m,
    since the custom_m will be used to store the Linear modules.

    If projection size is more than zero, then custom_m will contain
    a Sequential[Linear, Linear], otherwise it will contain a Linear.

    Parameters
    ----------
    input_size: int
        input size of SRU
    hidden_size: int
        hidden size of SRU
    num_layers: int
        number of layers in SRU
    projection_size: int
        size of projection
        if 0, then no projection
    linear_first_layer: bool
        if True, then use Linear for first layer, even if
        projection_size > 0
    custom_m: None
        must be None
    **sru_kwargs: Dict[str, Any]
        additional SRU kwargs
    """
    if custom_m is not None:
        raise Exception('custom_m must be None when using create_sru_using_linears')
    custom_m_list: List[nn.Module] = []

    out_features = (
        hidden_size * 3 if hidden_size == input_size else hidden_size * 4
    )
    if projection_size == 0:
        # create a list of nn.Linear() if projection trick is
        # not enabled
        custom_m_list = [nn.Linear(input_size, out_features, bias=False)]
        for i in range(num_layers - 1):
            custom_m_list.append(
                nn.Linear(hidden_size, hidden_size * 3, bias=False)
            )
    else:
        # create a list of nn.Sequantial() applying the
        # projection trick
        if not linear_first_layer:
            first_module: Union[nn.Sequential, nn.Linear] = nn.Sequential(
                nn.Linear(input_size, projection_size, bias=False),
                nn.Linear(projection_size, out_features, bias=False),
            )
        else:
            first_module = nn.Linear(
                input_size, out_features, bias=False)
        custom_m_list = [first_module]
        for i in range(num_layers - 1):
            custom_m_list.append(
                nn.Sequential(
                    nn.Linear(hidden_size, projection_size, bias=False),
                    nn.Linear(projection_size, hidden_size * 3, bias=False),
                )
            )
    sru = SRU(
        input_size,
        hidden_size,
        num_layers=num_layers,
        projection_size=projection_size,
        custom_m=custom_m_list,
        **sru_kwargs)
    return sru
