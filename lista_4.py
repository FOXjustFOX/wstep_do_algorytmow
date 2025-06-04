# Zadanie 1: Implementacja BFS, DFS oraz wariantów DFS


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class Queue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def enqueue(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def dequeue(self):
        if self.head is None:
            return None
        value = self.head.value
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        self.size -= 1
        return value

    def is_empty(self):
        return self.head is None


class Stack:
    def __init__(self):
        self.top = None
        self.size = 0

    def push(self, value):
        new_node = Node(value)
        new_node.next = self.top
        self.top = new_node
        self.size += 1

    def pop(self):
        if self.top is None:
            return None
        value = self.top.value
        self.top = self.top.next
        self.size -= 1
        return value

    def is_empty(self):
        return self.top is None


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# ukradzione
def print_tree_ascii(node, prefix="", is_left=True):
    if node is not None:
        print_tree_ascii(node.right, prefix + ("│   " if is_left else "    "), False)
        print(prefix + ("└── " if is_left else "┌── ") + str(node.value))
        print_tree_ascii(node.left, prefix + ("    " if is_left else "│   "), True)


def zad_1():

    print("----------------------------------------------------")
    print("Zadanie 1: Implementacja BFS, DFS oraz wariantów DFS")
    print("----------------------------------------------------")

    def bfs(root):
        if not root:
            return []
        queue = Queue()
        queue.enqueue(root)
        result = []
        while not queue.is_empty():
            node = queue.dequeue()
            result.append(node.value)
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)
        return result

    def dfs_preorder(root):
        result = []
        stack = Stack()
        stack.push(root)
        while not stack.is_empty():
            node = stack.pop()
            if node:
                result.append(node.value)
                stack.push(node.right)
                stack.push(node.left)
        return result

    def dfs_inorder(root):
        result = []
        stack = Stack()
        current = root
        while True:
            if current:
                stack.push(current)
                current = current.left
            elif not stack.is_empty():
                current = stack.pop()
                result.append(current.value)
                current = current.right
            else:
                break
        return result

    def dfs_postorder(root):
        result = []
        stack = Stack()
        stack.push(root)
        prev = None
        while not stack.is_empty():
            current = stack.top.value if stack.top else None
            if prev is None or prev.left == current or prev.right == current:
                if current.left:
                    stack.push(current.left)
                elif current.right:
                    stack.push(current.right)
            elif current.left == prev:
                if current.right:
                    stack.push(current.right)
            else:
                result.append(current.value)
                stack.pop()
            prev = current
        return result

    # Przykładowe drzewo
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)

    print("BFS:", bfs(root))
    print("DFS Preorder:", dfs_preorder(root))
    print("DFS Inorder:", dfs_inorder(root))
    print("DFS Postorder:", dfs_postorder(root))


def zad_2_2():

    print("----------------------------------------------------")
    print("Zadanie 2: Tworzenie nowego drzewa z liścia")
    print("----------------------------------------------------")

    def make_new_tree_from_leaf(old_root, leaf_value):
        """Tworzy nowe drzewo z podanego liścia"""
        # Znajdowanie ścieżki do liścia
        path = []

        def find_path(node, target):
            nonlocal path
            if not node:
                return False
            path.append(node)
            if node.value == target:
                return True
            if find_path(node.left, target) or find_path(node.right, target):
                return True
            path.pop()
            return False

        if not find_path(old_root, leaf_value):
            return None

        # Budowa nowego drzewa
        new_root = TreeNode(leaf_value)
        current = new_root

        # Odwracanie ścieżki od liścia do korzenia
        for i in range(len(path) - 2, -1, -1):
            if path[i].left == path[i + 1]:
                current.left = TreeNode(path[i].value)
                current = current.left
            elif path[i].right == path[i + 1]:
                current.right = TreeNode(path[i].value)
                current = current.right

        return new_root

    def bfs(root):
        """Pomocnicza funkcja do wizualizacji drzewa"""
        if not root:
            return []

        result = []
        q = Queue()
        q.enqueue(root)

        while not q.is_empty():
            node = q.dequeue()
            result.append(node.value)

            if node.left:
                q.enqueue(node.left)
            if node.right:
                q.enqueue(node.right)

        return result

    # Przykładowe drzewo
    original_root = TreeNode(1)
    original_root.left = TreeNode(2)
    original_root.right = TreeNode(3)
    original_root.left.left = TreeNode(4)
    original_root.left.right = TreeNode(5)
    original_root.right.right = TreeNode(6)

    # Test - tworzymy nowe drzewo z liścia 4
    new_tree_root = make_new_tree_from_leaf(original_root, 4)

    print("Original:", bfs(original_root))
    print_tree_ascii(original_root)
    print("Z liścia:", bfs(new_tree_root) if new_tree_root else "Liść nie istnieje")
    # print("Nowe drzewo z liścia 4:")
    print_tree_ascii(new_tree_root)

def zad_2():
    print("----------------------------------------------------")
    print("Zadanie 2: Tworzenie nowego drzewa z liścia (restrukturyzacja)")
    print("----------------------------------------------------")

    def _find_path_nodes_recursive(current_node, target_value, path_accumulator):
        if not current_node:
            return False
        
        path_accumulator.append(current_node)
        
        if current_node.value == target_value:
            return True
        
        if _find_path_nodes_recursive(current_node.left, target_value, path_accumulator):
            return True
        
        if _find_path_nodes_recursive(current_node.right, target_value, path_accumulator):
            return True
        
        path_accumulator.pop() 
        return False

    def make_new_tree_from_leaf(old_root, leaf_value_to_be_new_root):

        if not old_root:
            return None

        path_nodes = []
        if not _find_path_nodes_recursive(old_root, leaf_value_to_be_new_root, path_nodes):
            print(f"Node with value {leaf_value_to_be_new_root} not found in the tree.")
            return old_root

        target_node_for_root = path_nodes[-1]

        if len(path_nodes) == 1: 
            return old_root

        current_top_of_transformed_path = target_node_for_root 
        
        for i in range(len(path_nodes) - 2, -1, -1):
            P = path_nodes[i]  
            C = current_top_of_transformed_path 
            
            if P.left == C: 
              
                original_C_right_subtree = C.right
                C.right = P                        
                P.left = original_C_right_subtree 
            
            elif P.right == C:

                original_C_left_subtree = C.left 
                C.left = P                       
                P.right = original_C_left_subtree
            else:

                print(f"Error: Structure mismatch during rotation. {P.value} is not parent of {C.value}.")
                return old_root 


            if i > 0:
                G = path_nodes[i-1] 
                if G.left == P:
                    G.left = C
                elif G.right == P:
                    G.right = C
              
        return current_top_of_transformed_path


    # Helper BFS for visualization within zad_2
    def bfs_local(root):
        if not root: return []
        result = []; q = Queue(); q.enqueue(root)
        while not q.is_empty():
            node = q.dequeue(); result.append(node.value)
            if node.left: q.enqueue(node.left)
            if node.right: q.enqueue(node.right)
        return result

    # Rebuild the original tree for the next test, as make_new_tree_from_leaf modifies it
    original_root = TreeNode(1)
    original_root.left = TreeNode(2)
    original_root.right = TreeNode(3)
    original_root.left.left = TreeNode(4)
    original_root.left.right = TreeNode(5)
    original_root.right.right = TreeNode(6)
    original_root.left.left.left = TreeNode(7)
    
    print("original:")
    print_tree_ascii(original_root)
    print("BFS: ", bfs_local(original_root))

    print("\nLiść 5")
    new_tree_root = make_new_tree_from_leaf(original_root, 6)
    print("New: ", bfs_local(new_tree_root))
    print_tree_ascii(new_tree_root)
    
    
def zad_3():

    print("----------------------------------------------------")
    print("Zadanie 3: Drzewo czteropoziomowe")
    print("----------------------------------------------------")

    def print_tree_levels(root):
        if not root:
            print("Puste drzewo")
            return
        queue = Queue()
        queue.enqueue((root, 0))
        current_level = -1
        while not queue.is_empty():
            node, level = queue.dequeue()
            if level != current_level:
                print(f"\nPoziom {level}:", end=" ")
                current_level = level
            print(node.value, end=" ")
            if node.left:
                queue.enqueue((node.left, level + 1))
            if node.right:
                queue.enqueue((node.right, level + 1))
        print()

    def count_levels(root):
        levels = {}
        queue = Queue()
        queue.enqueue((root, 0))
        while not queue.is_empty():
            node, level = queue.dequeue()
            if level not in levels:
                levels[level] = 0
            levels[level] += 1
            if node.left:
                queue.enqueue((node.left, level + 1))
            if node.right:
                queue.enqueue((node.right, level + 1))
        return levels

    def count_leaves(root):
        leaves = 0
        queue = Queue()
        queue.enqueue(root)
        while not queue.is_empty():
            node = queue.dequeue()
            if not node.left and not node.right:
                leaves += 1
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)
        return leaves

    def find_shortest_path_to_leaf(root):
        """Znajduje najkrótszą ścieżkę od korzenia do dowolnego liścia"""
        if not root:
            return []

        # Użyj BFS, aby znaleźć najkrótszą ścieżkę
        queue = Queue()
        # Przechowuj węzeł i ścieżkę do niego
        queue.enqueue((root, [root.value]))

        while not queue.is_empty():
            node, path = queue.dequeue()

            # Jeśli znaleziono liść, zwróć ścieżkę
            if not node.left and not node.right:
                return path

            # Dodaj do kolejki dzieci węzła wraz z ich ścieżkami
            if node.left:
                queue.enqueue((node.left, path + [node.left.value]))
            if node.right:
                queue.enqueue((node.right, path + [node.right.value]))

        return []  # Jeśli brak liści

    def print_shortest_path_to_leaf(root):
        """Drukuje najkrótszą ścieżkę od korzenia do liścia"""
        path = find_shortest_path_to_leaf(root)

        print("Najkrótsza ścieżka od korzenia do liścia:")
        print(" -> ".join(str(value) for value in path))

    # Drzewo czteropoziomowe
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.left.left.left = TreeNode(7)
    root.left.left.right = TreeNode(8)

    print("\nStruktura drzewa:")
    print_tree_ascii(root)
    print("Liczba węzłów na poziomach:", count_levels(root))
    print("Liczba liści:", count_leaves(root))

    print("\nDla oryginalnego drzewa:")
    print_shortest_path_to_leaf(root)


def convert_bst_to_avl(root):
    """Funkcja konwertująca BST na AVL"""

    sorted_nodes = []

    def inorder_collect(node):
        if not node:
            return
        inorder_collect(node.left)
        sorted_nodes.append(node.value)
        inorder_collect(node.right)

    inorder_collect(root)

    # Tworzenie zbalansowanego drzewa AVL
    def build_balanced_tree(nodes, start, end):
        if start > end:
            return None

        # Middle element as root
        mid = (start + end) // 2
        root = TreeNode(nodes[mid])

        # Lewy sub
        root.left = build_balanced_tree(nodes, start, mid - 1)

        # Prawy sub
        root.right = build_balanced_tree(nodes, mid + 1, end)

        return root

    if not sorted_nodes:
        return None

    return build_balanced_tree(sorted_nodes, 0, len(sorted_nodes) - 1)


def bst_to_avl_d_1():
    print("----------------------------------------------------")
    print("Converting BST to AVL Tree")
    print("----------------------------------------------------")

    # niezbalansowane
    bst_root = TreeNode(10)
    bst_root.left = TreeNode(5)
    bst_root.left.left = TreeNode(3)
    bst_root.left.left.left = TreeNode(1)

    def print_tree_levels(root):
        if not root:
            print("Empty tree")
            return
        queue = Queue()
        queue.enqueue((root, 0))
        current_level = -1
        while not queue.is_empty():
            node, level = queue.dequeue()
            if level != current_level:
                print(f"\nLevel {level}:", end=" ")
                current_level = level
            print(node.value, end=" ")
            if node.left:
                queue.enqueue((node.left, level + 1))
            if node.right:
                queue.enqueue((node.right, level + 1))
        print()

    print("Original BST:")
    print_tree_ascii(bst_root)

    # AVL
    avl_root = convert_bst_to_avl(bst_root)

    print("\nConverted AVL Tree:")
    print_tree_ascii(avl_root)


def bst_to_avl_d_2():
    print("----------------------------------------------------")
    print("Converting BST to AVL Tree")
    print("----------------------------------------------------")

    bst_root = TreeNode(5)
    bst_root.left = TreeNode(4)
    bst_root.right = TreeNode(6)
    bst_root.right.right = TreeNode(7)
    bst_root.right.right.right = TreeNode(8)
    bst_root.left.left = TreeNode(3)
    bst_root.left.left.left = TreeNode(1)

    def print_tree_levels(root):
        if not root:
            print("Empty tree")
            return
        queue = Queue()
        queue.enqueue((root, 0))
        current_level = -1
        while not queue.is_empty():
            node, level = queue.dequeue()
            if level != current_level:
                print(f"\nPoziom {level}:", end=" ")
                current_level = level
            print(node.value, end=" ")
            if node.left:
                queue.enqueue((node.left, level + 1))
            if node.right:
                queue.enqueue((node.right, level + 1))
        print()

    print("Original BST:")
    print_tree_ascii(bst_root)

    avl_root = convert_bst_to_avl(bst_root)

    # Display the AVL tree
    print("\nConverted AVL Tree:")
    print_tree_ascii(avl_root)


zad_1()
zad_2()
zad_3()
bst_to_avl_d_1()
bst_to_avl_d_2()
