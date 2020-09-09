package tree

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func CreateTree(Nums []int) (root *TreeNode) {
	root = new(TreeNode)

	if len(Nums) > 1 {
		treeNodes := make([]*TreeNode,0)
		root.Val = Nums[0]
		index := 0

		var f func(node *TreeNode)

		f = func(node *TreeNode) {
			if index++;index<len(Nums) {
				node.Left = &TreeNode{Val: Nums[index]}
				treeNodes = append(treeNodes,node.Left)
			}

			if index++;index<len(Nums) {
				node.Right = &TreeNode{Val: Nums[index]}
				treeNodes = append(treeNodes,node.Right)
			}

			if len(treeNodes) > 0 {
				t := treeNodes[0]
				treeNodes = treeNodes[1:]
				f(t)
			}
		}

		f(root)
	}

	if len(Nums) > 0 {
		root.Val = Nums[0]
	}

	return
}

func FrontPrint(root *TreeNode)  {
	if root == nil {
		return
	}
	println(root.Val)
	FrontPrint(root.Left)
	FrontPrint(root.Right)
}