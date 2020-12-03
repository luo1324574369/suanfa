package main

import (
	"fmt"
)

func main() {
	fmt.Println(isPowerOfTwo(16))
}


//*****************常用*************************//
func binarySearch(nums []int,target int) int {
	left,right := 0,len(nums)-1

	for left <= right {
		mid := left + (right - left) / 2
		if nums[mid] == target {
			return mid
		}else if nums[mid] > target {
			right = mid - 1
		}else if nums[mid] < target {
			left = mid + 1
		}
	}

	return -1
}

func leftBound(nums []int,target int) int {
	left,right := 0,len(nums)-1

	for left <= right {
		mid := left + (right - left) / 2
		if nums[mid] == target {
			right = mid - 1
		}else if nums[mid] > target {
			right = mid - 1
		}else if nums[mid] < target {
			left = mid + 1
		}
	}

	if left >= len(nums) {
		return len(nums)
	}

	return left
}

func rightBound(nums []int,target int) int {
	left,right := 0,len(nums)-1

	for left <= right {
		mid := left + (right - left) / 2
		if nums[mid] == target {
			left = mid + 1
		}else if nums[mid] > target {
			right = mid - 1
		}else if nums[mid] < target {
			left = mid + 1
		}
	}

	if right < 0 || nums[right] != target {
		return -1
	}

	return right
}

func reverse(target []int,x int,y int){
	for x < y {
		target[x],target[y] = target[y],target[x]
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

//*****************队列*************************//
type DQueue struct {
	data []int
	len int
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

func (dq *DQueue) PushBack(n int)  {
	dq.data = append(dq.data,n)
	dq.len++
}

func (dq *DQueue) Push(n int)  {
	for dq.Len() > 0 && dq.Back() < n {
		dq.PopBack()
	}

	dq.PushBack(n)
}

func (dq *DQueue) Pop(n int) int {
	if dq.Len() > 0  && dq.Front() == n{
		return dq.PopFront()
	}

	return 0
}

//*****************链表*************************//
type ListNode struct {
	Val  int
	Next *ListNode
}

func createListNode(nums []int) *ListNode {
	list := new(ListNode)
	temp := list
	for _,v := range nums {
		temp.Next = &ListNode{Val: v}
		temp = temp.Next
	}

	return list.Next
}

func printListList(nodeList []*ListNode)  {
	for i:=0;i<len(nodeList);i++{
		fmt.Printf("\n--------------\n")
		printListNodes(nodeList[i])
	}
}

func printListNodes(node *ListNode)  {
	if node != nil {
		fmt.Printf("val:%d \n",node.Val)
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

//****************树***************************//
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Node struct {
	Val int
	Children []*Node
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
			if index++;index<len(Nums) && Nums[index] != -1 {
				node.Left = &TreeNode{Val: Nums[index]}
				treeNodes = append(treeNodes,node.Left)
			}

			if index++;index<len(Nums) && Nums[index] != -1 {
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

//前序遍历树
func frontPrintTree(root *TreeNode) {
	if root != nil {
		println(root.Val)
		if root.Left != nil {
			frontPrintTree(root.Left)
		}
		if root.Right != nil {
			frontPrintTree(root.Right)
		}
	}
}
