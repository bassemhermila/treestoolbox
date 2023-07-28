from classes.treeclasses import Tree

file_path = 'treestoolbox/sample/sample_tree.swc'
intree = Tree()
intree.load(file_path)
tree = intree.tran()
print(tree.X)