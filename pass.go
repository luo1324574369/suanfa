package main

import (
	"math"
	"sort"
)

//打家劫舍 II
//https://leetcode-cn.com/problems/house-robber-ii/
func rob2(nums []int) int {
	nl := len(nums)

	if nl == 1 {
		return nums[0]
	}

	var r func(start int,end int) int

	r = func(start int, end int) int {
		pre1 := 0
		pre2 := 0
		for i:=end;i>=start;i--{
			temp := pre1
			pre1 = max(pre1, pre2+ nums[i])
			pre2 = temp
		}

		return pre1
	}

	return max(r(0,nl-2),r(1,nl-1))
}

//打家劫舍
//https://leetcode-cn.com/problems/house-robber/
func rob1(nums []int) int {
	nl := len(nums)

	pre1 := 0
	pre2 := 0
	for i:=nl-1;i>=0;i--{
		temp := pre1
		pre1 = max(pre1, pre2+ nums[i])
		pre2 = temp
	}

	return pre1
}

//买买卖股票的最佳时机 IV
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/
func maxProfit4(k int, prices []int) int {
	maxK := k
	pl := len(prices)
	dp := make([][][2]int, pl)

	if pl == 0 {
		return 0
	}

	if k > pl/2 {
		return maxProfit2(prices)
	}

	for i := 0; i < pl; i++ {
		for j := 0; j <= maxK; j++ {
			dp[i] = append(dp[i], [2]int{})
		}
	}

	for i := 0; i < pl; i++ {
		for j := 1; j <= maxK; j++ {
			if i == 0 {
				dp[i][j][0] = 0
				dp[i][j][1] = -prices[i]
				continue
			}

			dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])
			dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])
		}
	}

	return dp[pl-1][maxK][0]
}

//买卖股票的最佳时机 III
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/
func maxProfit3(prices []int) int {
	maxK := 2
	pl := len(prices)
	dp := make([][3][2]int, pl)

	for i := 0; i < pl; i++ {
		dp[i] = [3][2]int{}
	}

	for i := 0; i < pl; i++ {
		for j := 1; j <= maxK; j++ {
			if i == 0 {
				dp[i][j][0] = 0
				dp[i][j][1] = -prices[i]
				continue
			}

			dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])
			dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])
		}
	}

	return dp[pl-1][maxK][0]
}

//买卖股票的最佳时机 II
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
func maxProfit2(prices []int) int {
	pre_has := math.MinInt32
	pre_no_has := 0

	for i := 0; i < len(prices); i++ {
		temp := pre_has
		pre_has = max(pre_has, pre_no_has-prices[i])
		pre_no_has = max(pre_no_has, temp+prices[i])
	}

	return pre_no_has
}

//买卖股票的最佳时机
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
func maxProfit1(prices []int) int {
	pre_has := math.MinInt32
	pre_no_has := 0

	for i := 0; i < len(prices); i++ {
		pre_no_has = max(pre_no_has, pre_has+prices[i])
		pre_has = max(pre_has, -prices[i])
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
	for i := 1; i < li; i++ {
		if intervals[i][0] < end {
			res++
		} else {
			end = intervals[i][1]
		}
	}

	return res
}

// N叉树的前序遍历
// https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/submissions/
func preorder(root *Node) []int {
	res := make([]int, 0)

	var front func(root *Node)
	front = func(root *Node) {
		if root != nil {
			res = append(res, root.Val)
			for _, v := range root.Children {
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
	treeNodes := make([]*TreeNode, 0)
	listNodes := make([]*ListNode, 0)

	if tree == nil {
		return listNodes
	}

	treeNodes = append(treeNodes, tree)

	for len(treeNodes) > 0 {
		nextTreeNodes := make([]*TreeNode, 0)
		tmpNode := &ListNode{}
		headNode := tmpNode

		for i := 0; i < len(treeNodes); i++ {
			tmpNode.Next = &ListNode{Val: treeNodes[i].Val}
			tmpNode = tmpNode.Next

			if treeNodes[i].Left != nil {
				nextTreeNodes = append(nextTreeNodes, treeNodes[i].Left)
			}
			if treeNodes[i].Right != nil {
				nextTreeNodes = append(nextTreeNodes, treeNodes[i].Right)
			}
		}

		listNodes = append(listNodes, headNode.Next)
		treeNodes = nextTreeNodes
	}

	return listNodes
}
