package main

import "sort"


//买卖股票的最佳时机
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
func maxProfit1(prices []int) int {
	pre_has := ^int(^uint(0) >> 1)
	pre_no_has := 0

	for i:=0;i<len(prices);i++{
		pre_no_has = max(pre_no_has,pre_has+prices[i])
		pre_has = max(pre_has,-prices[i])
	}

	return pre_no_has
}

//无重叠区间
//https://leetcode-cn.com/problems/non-overlapping-intervals/
func eraseOverlapIntervals(intervals [][]int) int {
	li := len(intervals)
	res := 0
	if li == 0 {
		return res
	}

	//冒泡排序
	//for i:=0;i<li;i++{
	//	for j:=0;j<li-i-1;j++{
	//		if intervals[j][1] > intervals[j+1][1] {
	//			intervals[j],intervals[j+1] = intervals[j+1],intervals[j]
	//		}
	//	}
	//}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][1] < intervals[j][1]
	})

	end := intervals[0][1]
	for i:=1;i<li;i++{
		if intervals[i][0] < end {
			res++
		}else{
			end = intervals[i][1]
		}
	}

	return res
}

// N叉树的前序遍历
// https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/submissions/
func preorder(root *Node) []int {
	res := make([]int,0)

	var front func(root *Node)
	front = func(root *Node) {
		if root != nil {
			res = append(res,root.Val)
			for _,v := range root.Children {
				front(v)
			}
		}
	}

	front(root)

	return res
}

// 特定深度节点链表
// https://leetcode-cn.com/problems/list-of-depth-lcci/submissions/
func listOfDepth(tree *TreeNode) []*ListNode {
	treeNodes := make([]*TreeNode,0)
	listNodes := make([]*ListNode,0)

	if tree == nil {
		return listNodes
	}

	treeNodes = append(treeNodes,tree)

	for len(treeNodes) > 0 {
		nextTreeNodes := make([]*TreeNode,0)
		tmpNode := &ListNode{}
		headNode := tmpNode

		for i:=0;i<len(treeNodes);i++{
			tmpNode.Next = &ListNode{Val: treeNodes[i].Val}
			tmpNode = tmpNode.Next

			if treeNodes[i].Left != nil {
				nextTreeNodes = append(nextTreeNodes,treeNodes[i].Left)
			}
			if treeNodes[i].Right != nil {
				nextTreeNodes = append(nextTreeNodes,treeNodes[i].Right)
			}
		}

		listNodes = append(listNodes, headNode.Next)
		treeNodes = nextTreeNodes
	}

	return listNodes
}