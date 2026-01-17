"""
Passes to apply before transpiling to numpy
"""

from cuda.tile._ir import ir
from cuda.tile._ir.ops import (
    TileLoadTokenOrdered,
    TileStoreTokenOrdered,
    MakeToken,
    JoinTokens,
    TileLoad,
    TileStore,
    TileReduce,
    TileArgReduce,
    BindMethod,
    GetBoundSelf,
    Assign,
)
from cuda.tile._ir.type import TileTy


def simplify_for_numpy(func):
    _eliminate_bound(func)
    _eliminate_tokens(func)
    _add_keepdim(func)


def _eliminate_bound(func):
    """
    Eliminate bind_method and get_bound_self operations.

    This pass finds patterns like:
        v1 = bind_method(obj, func)
        v2 = get_bound_self(v1)
        v3 = tile_reshape(v2, ...)

    And replaces them with:
        v2 = obj
        v3 = tile_reshape(v2, ...)

    The key insight is that we need to track variable objects, not just variable names,
    because variable names may have been renumbered by earlier optimization passes.
    """
    # Find all bind_method and get_bound_self operations
    # Use dictionaries that map the result Var objects themselves
    bind_ops = {}  # result_var -> BindMethod operation
    get_bound_self_ops = {}  # result_var -> GetBoundSelf operation

    for op in func.traverse():
        if isinstance(op, BindMethod):
            bind_ops[op.result_vars[0]] = op
        elif isinstance(op, GetBoundSelf):
            get_bound_self_ops[op.result_vars[0]] = op

    # If no operations found, return early
    if not bind_ops and not get_bound_self_ops:
        return

    # Build a mapping from get_bound_self result Var to the object Var
    gbs_to_object = {}

    for gbs_result_var, gbs_op in get_bound_self_ops.items():
        bound_method_var = gbs_op.bound_method
        if bound_method_var in bind_ops:
            bind_op = bind_ops[bound_method_var]
            # Map get_bound_self result to bind_method object
            gbs_to_object[gbs_result_var] = bind_op.object

    # Build mappings for Assign operations
    # When we eliminate get_bound_self, we need to replace it with: result_var = object_var
    # When we eliminate bind_method, we don't need to insert anything (the result is no longer used)
    assign_to_insert = {}  # get_bound_self.result_var -> object_var

    for gbs_result_var, object_var in gbs_to_object.items():
        assign_to_insert[gbs_result_var] = object_var

    # Build a mapper to replace variables
    mapper = ir.Mapper(func.ctx, preserve_vars=True)

    # Set up the variable mappings
    for old_var, new_var in gbs_to_object.items():
        mapper.set_var(old_var, new_var)

    # Process the function
    def _eliminate_bound_in_block(block):
        new_ops = []
        for op in block:
            if isinstance(op, GetBoundSelf):
                # Replace get_bound_self with an assign operation
                if op.result_vars[0] in gbs_to_object:
                    object_var = gbs_to_object[op.result_vars[0]]
                    # Create an assign: result_var = object_var
                    result_var = op.result_vars[0]
                    assign_op = Assign(object_var, result_var, op.loc)
                    new_ops.append(assign_op)
                else:
                    # Keep this get_bound_self as-is (shouldn't happen)
                    for nested_block in op.nested_blocks:
                        _eliminate_bound_in_block(nested_block)
                    new_op = op.clone(mapper)
                    new_ops.append(new_op)

            elif isinstance(op, BindMethod):
                # Don't add bind_method operation
                # Its result is no longer needed after we insert the assign
                pass

            else:
                # Process nested blocks
                for nested_block in op.nested_blocks:
                    _eliminate_bound_in_block(nested_block)

                # Clone the operation with the mapper
                new_op = op.clone(mapper)
                new_ops.append(new_op)

        block[:] = new_ops

    _eliminate_bound_in_block(func)


def _add_keepdim(block: ir.Block):
    """
    Add keepdims to the operation if it's a reduction operation.
    If the ndim of input are the same as the ndim of output, add keepdims=True;
    otherwise, add keepdims=False.
    """

    for op in block:
        if isinstance(op, TileReduce) or isinstance(op, TileArgReduce):
            x = op.x
            result_var = op.result_var

            x_type = x.get_type()
            res_type = result_var.get_type()

            if isinstance(x_type, TileTy) and isinstance(res_type, TileTy):
                keepdims = x_type.ndim == res_type.ndim
                op.attributes["keepdims"] = keepdims

        # Process nested blocks
        for nested_block in op.nested_blocks:
            _add_keepdim(nested_block)


def _eliminate_tokens(block: ir.Block):
    """Eliminate token operations in a block and its nested blocks."""
    new_ops = []

    for op in block:
        if isinstance(op, MakeToken):
            # Skip make_token operations entirely
            pass

        elif isinstance(op, TileLoadTokenOrdered):
            # Convert tile_load_token_ordered to tile_load
            # The token_ordered version returns (result, token), but we only need result
            result_var = op.result_vars[0]
            new_op = TileLoad(
                array=op.array,
                index=op.index,
                order=op.order,
                padding_mode=op.padding_mode,
                latency=op.latency,
                allow_tma=op.allow_tma,
                result_var=result_var,
                loc=op.loc,
            )
            new_ops.append(new_op)

        elif isinstance(op, TileStoreTokenOrdered):
            # Convert tile_store_token_ordered to tile_store
            # The token_ordered version takes a token and returns a token, but we ignore both
            new_op = TileStore(
                array=op.array,
                index=op.index,
                tile=op.tile,
                order=op.order,
                latency=op.latency,
                allow_tma=op.allow_tma,
                loc=op.loc,
            )
            new_ops.append(new_op)

        elif isinstance(op, JoinTokens):
            # Skip join_tokens operations entirely
            pass

        else:
            # Process nested blocks
            for nested_block in op.nested_blocks:
                _eliminate_tokens(nested_block)

            # Keep other operations as-is
            new_ops.append(op)

    block[:] = new_ops
