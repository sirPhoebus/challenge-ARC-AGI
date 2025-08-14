from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from app.utils import tokens as T
from app import config

# Program is a tree with ops and integer arguments (tokenized)

@dataclass(frozen=True)
class Node:
    op_tok: int
    args: Tuple[int, ...]  # for terminals (numbers/colors) or nested encoded by subprogram tokens

    def key(self) -> Tuple:
        return (self.op_tok, tuple(self.args))


class Program:
    def __init__(self, tokens: List[int]):
        self.tokens = tokens

    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __eq__(self, other) -> bool:
        return isinstance(other, Program) and self.tokens == other.tokens

    def __len__(self) -> int:
        return len(self.tokens)


def is_op(tok: int) -> bool:
    return tok in T.OP_ARITY


def arity(tok: int) -> int:
    return T.OP_ARITY.get(tok, 0)


def ast_to_tokens(ast: Tuple) -> List[int]:
    # ast format: (op_tok, [child_or_arg, ...]) with length = 1 + arity
    op = ast[0]
    children = ast[1]
    seq: List[int] = [op]
    for ch in children:
        if isinstance(ch, tuple):
            seq.extend(ast_to_tokens(ch))
        else:
            seq.append(int(ch))
    return seq


def tokens_to_ast(seq: List[int]) -> Optional[Tuple]:
    # Preorder parsing using arity
    i = 0

    def parse_at() -> Optional[Tuple]:
        nonlocal i
        if i >= len(seq):
            return None
        tok = seq[i]
        i += 1
        k = arity(tok)
        if k == 0:
            # number/color terminal used as op? not allowed
            return None
        children: List = []
        for _ in range(k):
            if i >= len(seq):
                return None
            nxt = seq[i]
            if is_op(nxt):
                ch = parse_at()
                if ch is None:
                    return None
                children.append(ch)
            else:
                children.append(nxt)
                i += 1
        return (tok, children)

    ast = parse_at()
    if ast is None or i != len(seq):
        return None
    return ast


def canonical_tokens(seq: List[int]) -> List[int]:
    # For now, identity. Placeholder for future canonicalization.
    return list(seq)


def valid_tokens(seq: List[int], max_depth: Optional[int] = None) -> bool:
    max_depth = max_depth if max_depth is not None else int(config.DSL_MAX_DEPTH)
    # Simple depth/arity check by parsing
    ast = tokens_to_ast(seq)
    if ast is None:
        return False

    def depth(t: Tuple) -> int:
        op, children = t
        d = 1
        for ch in children:
            if isinstance(ch, tuple):
                d = max(d, 1 + depth(ch))
        return d

    return depth(ast) <= max_depth
