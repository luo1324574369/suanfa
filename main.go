package main

import (
	"fmt"
)

func main() {
	frontPrintTree(constructMaximumBinaryTree2([]int{3, 2, 1, 6, 0, 5}))
}

// *****************常用*************************//

type PriorityQueue [][]int

func (p PriorityQueue) Len() int {
	return len(p)
}
func (p PriorityQueue) Less(i, j int) bool {
	return (p[i][0] + p[i][1]) < (p[j][0] + p[j][1])
}
func (p PriorityQueue) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}
func (p *PriorityQueue) Push(x interface{}) {
	*p = append(*p, x.([]int))
}
func (p *PriorityQueue) Pop() interface{} {
	old := *p
	n := len(old)
	item := old[n-1]
	*p = old[0 : n-1]
	return item
}

func binarySearch(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		}
	}

	return -1
}

func leftBound(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			right = mid - 1
		} else if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		}
	}

	if left >= len(nums) {
		return len(nums)
	}

	return left
}

func rightBound(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		}
	}

	if right < 0 || nums[right] != target {
		return -1
	}

	return right
}

func reverse(target []int, x int, y int) {
	for x < y {
		target[x], target[y] = target[y], target[x]
		x++
		y--
	}
}

func max(x int, y int) int {
	if x > y {
		return x
	}
	return y
}

func min(x int, y int) int {
	if x < y {
		return x
	}
	return y
}

func abs(x int) int {
	if x > 0 {
		return x
	}

	return -x
}

// *****************队列*************************//
type DQueue struct {
	data []int
	len  int
}

func (dq *DQueue) PopFront() (res int) {
	res = dq.data[0]
	dq.data = dq.data[1:]
	dq.len--
	return
}

func (dq *DQueue) PopBack() (res int) {
	dql := dq.Len()
	res = dq.data[dql-1]
	dq.data = dq.data[:dql-1]
	dq.len--
	return
}

func (dq *DQueue) Back() int {
	return dq.data[dq.Len()-1]
}

func (dq *DQueue) Front() int {
	return dq.data[0]
}

func (dq *DQueue) Len() int {
	return dq.len
}

func (dq *DQueue) PushBack(n int) {
	dq.data = append(dq.data, n)
	dq.len++
}

func (dq *DQueue) Push(n int) {
	for dq.Len() > 0 && dq.Back() < n {
		dq.PopBack()
	}

	dq.PushBack(n)
}

func (dq *DQueue) Pop(n int) int {
	if dq.Len() > 0 && dq.Front() == n {
		return dq.PopFront()
	}

	return 0
}

// *****************链表*************************//
type ListNode struct {
	Val  int
	Next *ListNode
}

func createListNode(nums []int) *ListNode {
	list := new(ListNode)
	temp := list
	for _, v := range nums {
		temp.Next = &ListNode{Val: v}
		temp = temp.Next
	}

	return list.Next
}

func printListList(nodeList []*ListNode) {
	for i := 0; i < len(nodeList); i++ {
		fmt.Printf("\n--------------\n")
		printListNodes(nodeList[i])
	}
}

func printListNodes(node *ListNode) {
	if node != nil {
		fmt.Printf("val:%d \n", node.Val)
		printListNodes(node.Next)
	}
}

//type DoubleNode struct {
//	Key int
//	Value int
//	Next *DoubleNode
//	Pre *DoubleNode
//}
//
//type DoubleList struct {
//	Head *DoubleNode
//	Tail *DoubleNode
//	Len int
//}
//
//func DList() *DoubleList {
//	dl := &DoubleList{}
//	dl.Head = &DoubleNode{Key: 0,Value: 0}
//	dl.Tail = &DoubleNode{Key: 0,Value: 0}
//	dl.Head.Next = dl.Tail
//	dl.Tail.Pre = dl.Head
//	return dl
//}
//
//func (dl *DoubleList) Append(d *DoubleNode)  {
//	d.Next = dl.Tail
//	d.Pre = dl.Tail.Pre
//
//	dl.Tail.Pre.Next = d
//	dl.Tail.Pre = d
//	dl.Len++
//}
//
//func (dl *DoubleList) Delete(d *DoubleNode)  {
//	d.Pre.Next = d.Next
//	d.Next.Pre = d.Pre
//	dl.Len--
//}
//
//func (dl *DoubleList) DeleteHead() *DoubleNode {
//	if dl.Head.Next == dl.Tail {
//		return nil
//	}
//
//	tmp := dl.Head.Next
//	dl.Delete(tmp)
//	return tmp
//}

// ****************树***************************//
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Node struct {
	Val      int
	Children []*Node
}

// 根据一个int切片,创建一个二叉树
func CreateTree(Nums []int) *TreeNode {
	if len(Nums) == 0 {
		return nil
	}

	root := &TreeNode{
		Val: Nums[0],
	}
	treeNodes := make([]*TreeNode, 0)
	treeNodes = append(treeNodes, root)

	index := 0
	for len(treeNodes) != 0 {
		node := treeNodes[0]
		treeNodes = treeNodes[1:]

		index++
		if index < len(Nums) && Nums[index] != -1 {
			left := &TreeNode{Val: Nums[index]}
			node.Left = left
			treeNodes = append(treeNodes, left)
		}
		index++
		if index < len(Nums) && Nums[index] != -1 {
			right := &TreeNode{Val: Nums[index]}
			node.Right = right
			treeNodes = append(treeNodes, right)
		}
	}
	return root
}

// 前序遍历树
func frontPrintTree(root *TreeNode) {
	if root == nil {
		return
	}
	println(root.Val)
	frontPrintTree(root.Left)
	frontPrintTree(root.Right)
}
