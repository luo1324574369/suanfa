package main

import "fmt"

func main() {
	r := CreateTree([]int{1,2,3,4,5})
	res := listOfDepth(r)

	fmt.Println(res)
}

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func listOfDepth(tree *TreeNode) []*ListNode {
	treeNodes := make([]*TreeNode,0)
	listNodes := make([]*ListNode,0)

	if tree == nil {
		return listNodes
	}

	treeNodes = append(treeNodes,tree)
	var depthQ func([]*TreeNode)

	depthQ = func(nodes []*TreeNode) {
		nextTreeNodes := make([]*TreeNode,0)

		oneNode := new(ListNode)
		oneNode.Val = nodes[0].Val

		for i:=1;i<len(nodes);i++{
			oneNode.Next = &ListNode{Val: nodes[i].Val}

			if nodes[i].Left != nil {
				nextTreeNodes = append(nextTreeNodes,nodes[i].Left)
			}

			if nodes[i].Right != nil {
				nextTreeNodes = append(nextTreeNodes,nodes[i].Right)
			}
		}

		listNodes = append(listNodes,oneNode)
		if len(nextTreeNodes) > 0 {
			depthQ(nextTreeNodes)
		}
	}

	depthQ(treeNodes)

	return listNodes
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
