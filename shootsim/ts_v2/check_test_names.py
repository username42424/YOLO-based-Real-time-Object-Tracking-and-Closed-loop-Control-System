# -*- coding: utf-8 -*-
"""Check for duplicate test method names across the tests directory."""
import ast
import os
import sys

TESTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tests")
seen = {}
dups = []
for fn in sorted(os.listdir(TESTS)):
    if not fn.startswith("test_") or not fn.endswith(".py"):
        continue
    path = os.path.join(TESTS, fn)
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (isinstance(item, ast.FunctionDef)
                        and item.name.startswith("test")):
                    key = f"{fn}.{node.name}.{item.name}"
                    if item.name in seen:
                        dups.append((item.name, seen[item.name], key))
                    seen.setdefault(item.name, key)
print("total test methods:", len(seen))
if dups:
    print("DUPLICATES:")
    for d in dups:
        print("  ", d)
    sys.exit(1)
print("no duplicate test method names")
