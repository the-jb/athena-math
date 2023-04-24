from collections import Counter
import itertools


class Op:
    NUM = "NUM"
    CON = "CON"
    NEG = "NEG"
    INV = "INV"
    ADD = "ADD"
    MUL = "MUL"
    ROOT = "ROOT"
    POW = "POW"
    HALF = "HALF"


class ThoughtObject:
    OPS = ()
    expr: str
    values: dict

    def __init__(self, op, *params):
        assert op in self.OPS
        self.op = op
        self.params = params
        self._count_params = Counter(params)
        self._set_params = frozenset(self._count_params.items())

    @property
    def depth(self):
        raise NotImplementedError()

    @property
    def value(self):
        try:
            return eval(self.expr, self.values.copy())
        except:
            return 0

    @property
    def children(self):
        return self.params

    def __str__(self):
        return self.expr

    def __repr__(self):
        if len(self.params) > 1:
            return f"{self.op}{self.params}"
        else:
            return f"{self.op}({repr(self.params[0])})"

    def __eq__(self, other):
        return isinstance(other, ThoughtObject) and self.op == other.op and self._set_params == other._set_params

    def __hash__(self):
        return hash((self.op, self._set_params))

    def __contains__(self, item):
        assert isinstance(item, ThoughtObject)
        if self.op == item.op:
            if not (item._count_params - self._count_params):
                return True
        return any(item in param for param in self.params if isinstance(param, ThoughtObject))

    def __add__(self, other):
        return AggregateObject(Op.ADD, self, other)

    def __radd__(self, other):
        return AggregateObject(Op.ADD, other, self)

    def __neg__(self):
        return TransformObject(Op.NEG, self)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, ConstObject) and other.value in (1, -1, 2, -2):
            other = other.value
        if other == 0.5:
            return self / 2
        if other == -0.5:
            return -(self / 2)
        if other == 1:
            return self
        if other == -1:
            return -self
        if other == 2:
            return AggregateObject(Op.ADD, self, self)
        if other == -2:
            return -AggregateObject(Op.ADD, self, self)
        return AggregateObject(Op.MUL, self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if other == 2 or (isinstance(other, ConstObject) and other.value == 2):
            return TransformObject(Op.HALF, self)
        return self * TransformObject(Op.INV, other)

    def __rtruediv__(self, other):
        return other * TransformObject(Op.INV, self)

    def __pow__(self, power, modulo=None):
        if isinstance(power, ConstObject) and power.value in (0.5, 1, 2, 3):
            power = power.value
        if power == 1:
            return self
        if power == 2:
            return self * self
        if power == 3:
            return self * self * self
        if power == 0.5:
            return TransformObject(Op.ROOT, self)
        return OrderedAggrObject(Op.POW, self, power)

    def __rpow__(self, other):
        if not isinstance(other, ThoughtObject):
            return ConstObject(other) ** self


class NumberObject(ThoughtObject):
    OPS = (Op.NUM,)

    def __init__(self, idx, value):
        super().__init__(Op.NUM, idx)
        self.expr = f"n{idx}"
        self._value = value
        self.values = {self.expr: value}

    @property
    def depth(self):
        return 0

    @property
    def children(self):
        return ()

    @property
    def value(self):
        return self._value

    @property
    def idx(self):
        return self.params[0]


class ConstObject(ThoughtObject):
    OPS = (Op.CON,)

    def __init__(self, value):
        assert isinstance(value, (int, float))
        if int(value) == value:
            value = int(value)
        super().__init__(Op.CON, value)
        self.expr = f'x{str(value).replace(".", "_")}'
        self._value = value
        self.values = {self.expr: value}

    def __repr__(self):
        return str(self.value)

    def __mul__(self, other):
        if self.value in (1, -1, 2, -2) and isinstance(other, ThoughtObject):
            return self.value * other
        return super().__mul__(other)

    def __rtruediv__(self, other):
        if self.value == 1:
            return other
        if self.value == 2 and isinstance(other, ThoughtObject):
            return other / 2
        return super().__rtruediv__(other)

    @property
    def depth(self):
        return 0

    @property
    def children(self):
        return ()

    @property
    def value(self):
        return self._value


class TransformObject(ThoughtObject):
    OPS = (Op.NEG, Op.INV, Op.ROOT, Op.HALF)

    def __init__(self, op, param):
        if not isinstance(param, ThoughtObject):
            param = ConstObject(param) if param > 0 else -ConstObject(-param)
        self._depth = param.depth + 1
        super().__init__(op, param)
        p = param.expr if param.op in (Op.NUM, Op.CON) else f"( {param.expr} )"
        if op == Op.NEG:
            self.expr = f"-{p}"
        elif op == Op.INV:
            self.expr = f"1 / {p}"
        elif op == Op.ROOT:
            self.expr = f"{p} ** 0.5"
        elif op == Op.HALF:
            self.expr = f"{p} / 2"
        self.values = param.values

    @property
    def depth(self):
        return self._depth


class AggregateObject(ThoughtObject):
    OPS = (Op.ADD, Op.MUL)

    def __init__(self, op, param1, param2, *extras):
        if not isinstance(param1, ThoughtObject):
            param1 = ConstObject(param1) if param1 > 0 else -ConstObject(-param1)
        if not isinstance(param2, ThoughtObject):
            param2 = ConstObject(param2) if param2 > 0 else -ConstObject(-param2)
        params = (*param1.params,) if op == param1.op else (param1,)
        params += (*param2.params,) if op == param2.op else (param2,)
        params += extras
        self._children = (param1, param2)
        self._depth = max(param1.depth, param2.depth) + 1
        super().__init__(op, *params)

        if op == Op.ADD:
            expr = None
            for p in params:
                if expr is None:
                    expr = p.expr
                elif p.expr[0] == "-":
                    expr += " - " + p.expr[1:]
                else:
                    expr += " + " + p.expr
            self.expr = expr
        elif op == Op.MUL:
            expr = None
            for p in params:
                pex = p.expr if p.op in (Op.NUM, Op.CON, Op.INV) else f"( {p.expr} )"
                if expr is None:
                    expr = pex
                elif pex.startswith("1 / "):
                    expr += " / " + pex[4:]
                else:
                    expr += " * " + pex
            self.expr = expr
        self.values = param1.values | param2.values

    @property
    def depth(self):
        return self._depth

    @property
    def children(self):
        return self._children


class OrderedAggrObject(ThoughtObject):
    OPS = (Op.POW,)

    def __init__(self, op, param1, param2):
        if not isinstance(param1, ThoughtObject):
            param1 = ConstObject(param1) if param1 > 0 else -ConstObject(-param1)
        if not isinstance(param2, ThoughtObject):
            param2 = ConstObject(param2) if param2 > 0 else -ConstObject(-param2)
        self._depth = max(param1.depth, param2.depth) + 1
        super().__init__(op, param1, param2)
        p1 = param1.expr if param1.op in (Op.NUM, Op.CON) else f"( {param1.expr} )"
        p2 = (
            param2.value
            if param2.op == Op.CON and param2.value in (2, 3)
            else param2.expr
            if param2.op in (Op.NUM, Op.CON)
            else f"( {param2.expr} )"
        )
        if op == Op.POW:
            self.expr = f"{p1} ** {p2}"
            self.sub_op = ()
        self.values = param1.values | param2.values

    def __eq__(self, other):
        return isinstance(other, ThoughtObject) and self.op == other.op and self.params == other.params

    def __hash__(self):
        return hash((self.op, *self.params))

    @property
    def depth(self):
        return self._depth


class ThoughtExpander:
    def __init__(self, initial_thoughts, limit_depth, limit_thoughts=0, power=False):
        self.initial_thoughts = initial_thoughts
        self.limit_depth = limit_depth
        self.limit_thoughts = limit_thoughts
        self.power = power

    def __iter__(self):
        self.depth = 0
        self.n_transformed = 0
        self.n_aggregated = 0
        self.thoughts = []
        return self

    def __next__(self):
        if self.depth >= self.limit_depth > 0:
            raise StopIteration
        if self.limit_thoughts and len(self.thoughts) > self.limit_thoughts:
            raise StopIteration
        if self.depth == 0:
            results = self.initial_thoughts, None
        elif self.depth % 2 == 1:
            if len(self.thoughts) == self.n_transformed:
                raise StopIteration
            results = self._expand_thoughts_with_transform()
            self.n_transformed = len(self.thoughts)
        else:
            if len(self.thoughts) == self.n_aggregated:
                raise StopIteration
            results = self._expand_thoughts_with_aggregate()
            self.n_aggregated = len(self.thoughts)
        self.depth += 1
        return results

    def collect(self, thoughts):
        self.thoughts += thoughts

    def _expand_thoughts_with_transform(self):
        expanded_thoughts = []
        expanded_indices = []
        for op in TransformObject.OPS:
            for i, e in enumerate(self.thoughts[self.n_transformed :], self.n_transformed):
                if (thought := TransformObject(op, e)) not in self.thoughts:
                    expanded_thoughts.append(thought)
                    expanded_indices.append((op, i))
                    if self.limit_thoughts and len(self.thoughts) + len(expanded_thoughts) > self.limit_thoughts:
                        return expanded_thoughts, expanded_indices
        return expanded_thoughts, expanded_indices

    def _expand_thoughts_with_aggregate(self):
        existing_thoughts = self.thoughts[: self.n_aggregated]
        new_thoughts = self.thoughts[self.n_aggregated :]
        expanded_thoughts = []
        expanded_indices = []
        for op in AggregateObject.OPS:
            for (i, e1), (j, e2) in itertools.chain(
                (
                    ((i, e1), (j, e2))
                    for i, e1 in enumerate(existing_thoughts)
                    for j, e2 in enumerate(new_thoughts, self.n_aggregated)
                ),
                itertools.combinations_with_replacement(enumerate(new_thoughts, self.n_aggregated), 2),
            ):
                if (thought := AggregateObject(op, e1, e2)) not in self.thoughts and thought not in expanded_thoughts:
                    expanded_thoughts.append(thought)
                    expanded_indices.append((op, (i, j)))
                    if self.limit_thoughts and len(self.thoughts) + len(expanded_thoughts) > self.limit_thoughts:
                        return expanded_thoughts, expanded_indices

        if not self.power:
            return expanded_thoughts, expanded_indices

        for op in OrderedAggrObject.OPS:
            for i, e1 in enumerate(existing_thoughts):
                for j, e2 in enumerate(new_thoughts, self.n_aggregated):
                    if (
                        thought := OrderedAggrObject(op, e1, e2)
                    ) not in self.thoughts and thought not in expanded_thoughts:
                        expanded_thoughts.append(thought)
                        expanded_indices.append((op, (i, j)))
                    if (
                        thought := OrderedAggrObject(op, e2, e1)
                    ) not in self.thoughts and thought not in expanded_thoughts:
                        expanded_thoughts.append(thought)
                        expanded_indices.append((op, (j, i)))
                    if self.limit_thoughts and len(self.thoughts) + len(expanded_thoughts) > self.limit_thoughts:
                        return expanded_thoughts, expanded_indices
            for i, e1 in enumerate(new_thoughts, self.n_aggregated):
                for j, e2 in enumerate(new_thoughts, self.n_aggregated):
                    if (
                        thought := OrderedAggrObject(op, e1, e2)
                    ) not in self.n_aggregated and thought not in expanded_thoughts:
                        expanded_thoughts.append(thought)
                        expanded_indices.append((op, (i, j)))
                        if self.limit_thoughts and len(self.thoughts) + len(expanded_thoughts) > self.limit_thoughts:
                            return expanded_thoughts, expanded_indices

        return expanded_thoughts, expanded_indices
