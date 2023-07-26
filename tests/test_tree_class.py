from classes.treeclasses import Tree
import tools.edit as ed

file_path = 'treestoolbox/sample/sample_tree.swc'
intree = Tree()
intree.load(file_path)
tree = ed.tran(intree)
print(tree.X)