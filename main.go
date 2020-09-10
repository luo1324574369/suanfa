package main

import "fmt"

func main() {
	r := CreateTree([]int{})
	res := listOfDepth(r)

	printListList(res)
}

//算法


// 链表
type ListNode struct {
	Val  int
	Next *ListNode
}

func printListList(nodeList []*ListNode)  {
	for i:=0;i<len(nodeList);i++{
		fmt.Printf("\n--------------\n")
		printListNodes(nodeList[i])
	}
}

func printListNodes(node *ListNode)  {
	if node != nil {
		print(node.Val)
		print(",")
		printListNodes(node.Next)
	}
}
//树
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//根据一个int切片,创建一个二叉树
func CreateTree(Nums []int) (root *TreeNode) {
	root = new(TreeNode)

	if len(Nums) > 1 {
		treeNodes := make([]*TreeNode,0)
		root.Val = Nums[0]
		index := 0

		var f func(node *TreeNode)

		f = func(node *TreeNode) {
			if index++;index<len(Nums) && Nums[index] != 0 {
				node.Left = &TreeNode{Val: Nums[index]}
				treeNodes = append(treeNodes,node.Left)
			}

			if index++;index<len(Nums) && Nums[index] != 0 {
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
